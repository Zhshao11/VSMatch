#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002a.feature import load_slim_feature
from experiments.sparsegeo.e002b.candidates import indices_to_xy, matcher_core_topk, prepare_matcher_layout
from experiments.sparsegeo.e003a.candidate_cache import _load_image, _sample_truth_flow, read_flow
from experiments.sparsegeo.e003a.data_protocol import enforce_split_access, load_split_manifest
from experiments.sparsegeo.e004a.proposals import build_region_proposals
from experiments.sparsegeo.e004a.provenance import runtime_provenance, source_manifest


def exact_cosine_top8(source_feature, target_feature, source_indices, live_shapes=None, to_cpu=True):
    prepared = prepare_matcher_layout(
        source_feature, target_feature, source_indices,
        score_type="cosine", deployment_guard=False,
    )
    scores, indices = matcher_core_topk(
        prepared, candidate_count=8, target_chunk=1024, live_shapes=live_shapes,
    )
    values = (scores[0], indices[0], indices_to_xy(indices)[0])
    return tuple(value.cpu() for value in values) if to_cpu else values


def _candidate_table(source_feature, target_feature, source_indices, source_xy_token, live_shapes=None):
    scores, indices, target_xy = exact_cosine_top8(
        source_feature, target_feature, source_indices, live_shapes=live_shapes,
    )
    return {
        "source_lattice_indices": source_indices.to(torch.long).cpu(),
        "source_xy_token": source_xy_token.to(torch.long).cpu(),
        "source_xy_pixel": (8 * source_xy_token).to(torch.long).cpu(),
        "candidate_scores": scores,
        "candidate_indices": indices,
        "target_xy_token": target_xy,
    }


@torch.inference_mode()
def build_pair_candidates(model, optical, sar, live_shapes=None):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen GDROS model contains trainable parameters")
    source_feature, target_feature = model(optical, sar)
    if tuple(source_feature.shape) != (1, 256, 64, 64):
        raise RuntimeError(f"unexpected source feature shape: {tuple(source_feature.shape)}")
    proposals = build_region_proposals()
    public = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))
    proposal_shapes = [] if live_shapes is not None else None
    proposal = _candidate_table(
        source_feature, target_feature, proposals.lattice_indices,
        proposals.lattice_xy, proposal_shapes,
    )
    public_table = _candidate_table(
        source_feature, target_feature, public.lattice_indices, public.lattice_xy,
    )
    proposal.update({
        "proposal_id": proposals.proposal_id,
        "region_id": proposals.region_id,
        "offset_yx": proposals.offset_yx,
    })
    if live_shapes is not None:
        live_shapes.extend(proposal_shapes)
    return {"proposal_m256": proposal, "fixed_public_k128": public_table}


def build_table_labels(candidate_table, flow):
    source_pixel = candidate_table["source_xy_pixel"].to(torch.float32)
    target_token = candidate_table["target_xy_token"].to(torch.float32)
    target_pixel_gt = source_pixel + _sample_truth_flow(flow, source_pixel)
    target_token_gt = target_pixel_gt / 8.0
    in_bounds = (
        (target_pixel_gt[:, 0] >= 0.0) & (target_pixel_gt[:, 0] <= 511.0)
        & (target_pixel_gt[:, 1] >= 0.0) & (target_pixel_gt[:, 1] <= 511.0)
    )
    distances = torch.linalg.vector_norm(target_token - target_token_gt[:, None, :], dim=-1)
    nearest_error, nearest_slot = distances.min(dim=-1)
    return {
        "target_pixel_gt": target_pixel_gt.cpu(),
        "target_token_gt": target_token_gt.cpu(),
        "candidate_distance_gt": distances.cpu(),
        "nearest_candidate_error": nearest_error.cpu(),
        "nearest_candidate_slot": nearest_slot.cpu(),
        "in_bounds": in_bounds.cpu(),
        "recoverable": (in_bounds & (nearest_error <= 2.0)).cpu(),
    }


def build_pair_labels(candidates, flow):
    axis = torch.linspace(0.0, 511.0, 16)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return {
        "proposal_m256": build_table_labels(candidates["proposal_m256"], flow),
        "fixed_public_k128": build_table_labels(candidates["fixed_public_k128"], flow),
        "grid_xy_pixel": grid_xy,
        "grid_truth_flow": _sample_truth_flow(flow, grid_xy),
        "uses_raw_truth_flow": True,
        "uses_datum": False,
    }


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _paths(cache_root, split, pair_id):
    root = Path(cache_root)
    return root / "candidates" / split / f"{pair_id}.pt", root / "labels" / split / f"{pair_id}.pt"


def validate_cached_pair(candidate, labels, record, split, source_sha, checkpoint_sha):
    expected = {
        "pair_id": record["pair_id"],
        "split": split,
        "source_manifest_sha256": source_sha,
    }
    for name, value in expected.items():
        if candidate.get(name) != value or labels.get(name) != value:
            raise RuntimeError(f"cache identity mismatch for {record['pair_id']}: {name}")
    if candidate.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"cache checkpoint mismatch for {record['pair_id']}")
    if candidate.get("optical_sha256") != record["files"]["opt"]["sha256"]:
        raise RuntimeError(f"cache optical input mismatch for {record['pair_id']}")
    if candidate.get("sar_sha256") != record["files"]["sar_warped"]["sha256"]:
        raise RuntimeError(f"cache SAR input mismatch for {record['pair_id']}")
    if labels.get("truth_flow_sha256") != record["files"]["truth_flow"]["sha256"]:
        raise RuntimeError(f"cache truth-flow mismatch for {record['pair_id']}")
    if candidate.get("uses_ground_truth") is not False or candidate.get("uses_datum") is not False:
        raise RuntimeError("candidate cache GT/datum isolation violation")
    if labels.get("uses_raw_truth_flow") is not True or labels.get("uses_datum") is not False:
        raise RuntimeError("label cache authority/isolation violation")
    proposal, public = candidate.get("proposal_m256", {}), candidate.get("fixed_public_k128", {})
    proposal_labels, public_labels = labels.get("proposal_m256", {}), labels.get("fixed_public_k128", {})
    shape_checks = (
        proposal.get("candidate_scores", torch.empty(0)).shape == (256, 8),
        proposal.get("candidate_indices", torch.empty(0)).shape == (256, 8),
        proposal.get("source_xy_pixel", torch.empty(0)).shape == (256, 2),
        public.get("candidate_scores", torch.empty(0)).shape == (128, 8),
        public.get("candidate_indices", torch.empty(0)).shape == (128, 8),
        proposal_labels.get("recoverable", torch.empty(0)).shape == (256,),
        public_labels.get("recoverable", torch.empty(0)).shape == (128,),
        labels.get("grid_xy_pixel", torch.empty(0)).shape == (256, 2),
        labels.get("grid_truth_flow", torch.empty(0)).shape == (256, 2),
    )
    if not all(shape_checks):
        raise RuntimeError(f"cache schema mismatch for {record['pair_id']}")
    frozen = build_region_proposals()
    if not torch.equal(proposal["proposal_id"], frozen.proposal_id):
        raise RuntimeError(f"proposal ID mismatch for {record['pair_id']}")
    if not torch.equal(proposal["source_lattice_indices"], frozen.lattice_indices):
        raise RuntimeError(f"proposal lattice mismatch for {record['pair_id']}")
    tensor_groups = (proposal, public, proposal_labels, public_labels)
    if not all(
        torch.isfinite(value).all()
        for group in tensor_groups for value in group.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ):
        raise RuntimeError(f"nonfinite cache tensor for {record['pair_id']}")
    return True


def build_cache(args):
    provenance = runtime_provenance(
        args.spec_root, args.checkpoint, args.split_manifest, Path(__file__).resolve().parent,
    )
    source_sha = provenance["source"]["sha256"]
    records = load_split_manifest(args.split_manifest, allowed_split=args.split)
    selected = [row for index, row in enumerate(records) if index % args.shard_count == args.shard_index]
    if args.max_pairs is not None:
        selected = selected[:args.max_pairs]
    device = torch.device(args.device)
    model, _, _ = load_slim_feature(args.checkpoint, device)
    model.requires_grad_(False).eval()
    completed, traces = [], []
    for position, record in enumerate(selected):
        enforce_split_access(record, args.training_root, args.split)
        candidate_path, label_path = _paths(args.cache_root, args.split, record["pair_id"])
        if candidate_path.exists() or label_path.exists():
            if candidate_path.exists() and label_path.exists() and not args.overwrite:
                candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
                labels = torch.load(label_path, map_location="cpu", weights_only=True)
                validate_cached_pair(
                    candidate, labels, record, args.split, source_sha, provenance["checkpoint_sha256"],
                )
                completed.append(record["pair_id"])
                continue
            if not args.overwrite:
                raise RuntimeError(f"partial cache exists for {record['pair_id']}; refusing to overwrite")
        optical = _load_image(Path(record["files"]["opt"]["path"])).to(device)
        sar = _load_image(Path(record["files"]["sar_warped"]["path"])).to(device)
        live_shapes = []
        candidates = build_pair_candidates(model, optical, sar, live_shapes=live_shapes)
        candidate_record = {
            **candidates, "pair_id": record["pair_id"], "split": args.split,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "source_manifest_sha256": source_sha,
            "optical_sha256": record["files"]["opt"]["sha256"],
            "sar_sha256": record["files"]["sar_warped"]["sha256"],
            "uses_ground_truth": False, "uses_datum": False,
        }
        flow = read_flow(Path(record["files"]["truth_flow"]["path"]))
        label_record = {
            **build_pair_labels(candidates, flow), "pair_id": record["pair_id"],
            "split": args.split, "truth_flow_sha256": record["files"]["truth_flow"]["sha256"],
            "source_manifest_sha256": source_sha,
        }
        validate_cached_pair(
            candidate_record, label_record, record, args.split,
            source_sha, provenance["checkpoint_sha256"],
        )
        _atomic_torch_save(candidate_record, candidate_path)
        _atomic_torch_save(label_record, label_path)
        completed.append(record["pair_id"])
        traces.append({"pair_id": record["pair_id"], "live_shapes": live_shapes})
        if position % args.log_every == 0:
            print(f"P0 shard {args.shard_index}: {position}/{len(selected)} {record['pair_id']}", flush=True)
    result = {
        "status": "PASS", "stage": "P0_CACHE", "shard_index": args.shard_index,
        "shard_count": args.shard_count, "split": args.split, "pair_count": len(completed),
        "pair_ids": completed, "candidate_cache_uses_gt": False,
        "candidate_cache_uses_datum": False, "labels_use_raw_truth_flow": True,
        "provenance": provenance, "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "pair_count": len(completed), "output": str(args.output)}))


@torch.inference_mode()
def verify_cache_parity(args):
    provenance = runtime_provenance(
        args.spec_root, args.checkpoint, args.split_manifest, Path(__file__).resolve().parent,
    )
    records = load_split_manifest(args.split_manifest, allowed_split=args.split)[:args.pairs]
    device = torch.device(args.device)
    model, _, _ = load_slim_feature(args.checkpoint, device)
    model.requires_grad_(False).eval()
    rows = []
    for record in records:
        candidate_path, _ = _paths(args.cache_root, args.split, record["pair_id"])
        cached = torch.load(candidate_path, map_location="cpu", weights_only=True)
        optical = _load_image(Path(record["files"]["opt"]["path"])).to(device)
        sar = _load_image(Path(record["files"]["sar_warped"]["path"])).to(device)
        online = build_pair_candidates(model, optical, sar)
        checks = {}
        for variant in ("proposal_m256", "fixed_public_k128"):
            for key in ("source_lattice_indices", "source_xy_token", "candidate_indices", "target_xy_token"):
                checks[f"{variant}.{key}"] = bool(torch.equal(cached[variant][key], online[variant][key]))
            checks[f"{variant}.candidate_scores"] = bool(
                torch.equal(cached[variant]["candidate_scores"], online[variant]["candidate_scores"])
            )
        rows.append({"pair_id": record["pair_id"], "checks": checks, "pass": all(checks.values())})
    result = {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "provenance": provenance,
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    if result["status"] != "PASS":
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "parity"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--split-manifest", type=Path, required=True)
        sub.add_argument("--training-root", type=Path, required=True)
        sub.add_argument("--spec-root", type=Path, required=True)
        sub.add_argument("--cache-root", type=Path, required=True)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--split", choices=("train", "dev"), default="dev")
        sub.add_argument("--output", type=Path, required=True)
    build = subparsers.choices["build"]
    build.add_argument("--shard-count", type=int, default=1)
    build.add_argument("--shard-index", type=int, default=0)
    build.add_argument("--max-pairs", type=int)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--log-every", type=int, default=10)
    subparsers.choices["parity"].add_argument("--pairs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(args) if args.command == "build" else verify_cache_parity(args)
