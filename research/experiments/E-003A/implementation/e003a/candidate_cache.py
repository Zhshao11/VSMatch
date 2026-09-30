#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.utils import frame_utils
from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002a.feature import load_slim_feature
from experiments.sparsegeo.e002a.matcher import flatten_features
from experiments.sparsegeo.e002b.candidates import extract_candidates
from experiments.sparsegeo.e003a.data_protocol import (
    enforce_split_access,
    load_split_manifest,
    sha256_file,
)


CHECKPOINT_SHA256 = "80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683"
ANCHOR_SHA256 = "2872a818932f138d79a0a0aebf3fec2bbadb1939e839788f6f103c1e1f16efb6"
CANDIDATE_FEATURE_DIM = 13


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    else:
        array = array[..., :3]
    if array.shape != (512, 512, 3) or array.dtype != np.uint8:
        raise RuntimeError(f"E-003A requires decoded uint8 [512,512,3]: {path} -> {array.shape}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().unsqueeze(0)


def _pixel_normalize(xy_pixel: torch.Tensor) -> torch.Tensor:
    return 2.0 * xy_pixel.to(torch.float32) / 511.0 - 1.0


def build_candidate_features(source_feature, target_feature, candidate_set, anchor_set, to_cpu=True):
    if tuple(source_feature.shape) != (1, 256, 64, 64):
        raise RuntimeError(f"unexpected frozen feature shape: {tuple(source_feature.shape)}")
    scores = candidate_set.scores[0].to(torch.float32)
    indices = candidate_set.indices[0]
    target_xy = candidate_set.xy[0].to(torch.float32)
    if scores.shape != (128, 8) or indices.shape != (128, 8):
        raise RuntimeError("candidate table must be exactly [128,8]")

    source_tokens = flatten_features(source_feature)[0]
    target_tokens = flatten_features(target_feature)[0]
    source_indices = anchor_set.lattice_indices.to(source_feature.device)
    source_norm = torch.linalg.vector_norm(source_tokens.index_select(0, source_indices), dim=-1)
    target_norm = torch.linalg.vector_norm(target_tokens, dim=-1).gather(0, indices.reshape(-1)).view(128, 8)

    source_xy_token = anchor_set.lattice_xy.to(source_feature.device, torch.float32)
    source_xy_pixel = 8.0 * source_xy_token
    target_xy_pixel = 8.0 * target_xy
    source_xy_norm = _pixel_normalize(source_xy_pixel).unsqueeze(1).expand(-1, 8, -1)
    target_xy_norm = _pixel_normalize(target_xy_pixel)
    displacement = target_xy_norm - source_xy_norm

    rank = torch.arange(8, device=scores.device, dtype=torch.float32).view(1, 8).expand(128, -1) / 7.0
    relative_top1 = scores - scores[:, :1]
    gap = torch.zeros_like(scores)
    gap[:, :-1] = scores[:, :-1] - scores[:, 1:]
    multiplicity = torch.bincount(indices.reshape(-1), minlength=4096).to(torch.float32)
    multiplicity = multiplicity.gather(0, indices.reshape(-1)).view(128, 8) / 128.0

    features = torch.cat(
        (
            scores.unsqueeze(-1),
            rank.unsqueeze(-1),
            relative_top1.unsqueeze(-1),
            gap.unsqueeze(-1),
            source_xy_norm,
            target_xy_norm,
            displacement,
            torch.log1p(source_norm).view(128, 1, 1).expand(-1, 8, -1),
            torch.log1p(target_norm).unsqueeze(-1),
            multiplicity.unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape != (128, 8, CANDIDATE_FEATURE_DIM):
        raise RuntimeError(f"candidate feature shape mismatch: {tuple(features.shape)}")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("candidate feature contains NaN/Inf")
    result = {
        "candidate_features": features,
        "source_xy_token": source_xy_token,
        "source_xy_pixel": source_xy_pixel,
        "source_xy_norm": source_xy_norm[:, 0],
        "target_xy_token": target_xy,
        "target_xy_pixel": target_xy_pixel,
        "target_xy_norm": target_xy_norm,
        "candidate_scores": scores,
        "candidate_indices": indices,
    }
    return {key: value.cpu() for key, value in result.items()} if to_cpu else result


@torch.inference_mode()
def build_frozen_candidate_table(
    model, optical: torch.Tensor, sar: torch.Tensor, anchor_set, to_cpu=True, live_shapes=None
):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen GDROS model contains trainable parameters")
    source, target = model(optical, sar)
    candidate_set, _ = extract_candidates(
        source,
        target,
        anchor_set.lattice_indices,
        score_type="cosine",
        candidate_count=8,
        target_chunk=1024,
        live_shapes=live_shapes,
    )
    return build_candidate_features(source, target, candidate_set, anchor_set, to_cpu=to_cpu)


def read_flow(path: Path) -> torch.Tensor:
    array = np.asarray(frame_utils.read_gen(str(path)), dtype=np.float32)
    if array.shape != (512, 512, 2):
        raise RuntimeError(f"raw truth flow must be [512,512,2]: {path} -> {array.shape}")
    tensor = torch.from_numpy(array.copy())
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"raw truth flow contains NaN/Inf: {path}")
    return tensor


def _sample_truth_flow(flow: torch.Tensor, xy_pixel: torch.Tensor) -> torch.Tensor:
    if flow.shape != (512, 512, 2) or xy_pixel.shape[-1] != 2:
        raise ValueError("invalid truth-flow sampler input")
    grid = 2.0 * xy_pixel.to(torch.float32) / 511.0 - 1.0
    shape = grid.shape[:-1]
    sampled = F.grid_sample(
        flow.permute(2, 0, 1).unsqueeze(0),
        grid.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(2, -1).transpose(0, 1).reshape(*shape, 2)


def build_training_labels(candidate_table, flow: torch.Tensor):
    source_pixel = candidate_table["source_xy_pixel"].to(torch.float32)
    target_token = candidate_table["target_xy_token"].to(torch.float32)
    anchor_flow = _sample_truth_flow(flow, source_pixel)
    target_pixel_gt = source_pixel + anchor_flow
    target_token_gt = target_pixel_gt / 8.0
    in_bounds = (
        (target_pixel_gt[:, 0] >= 0.0)
        & (target_pixel_gt[:, 0] <= 511.0)
        & (target_pixel_gt[:, 1] >= 0.0)
        & (target_pixel_gt[:, 1] <= 511.0)
    )
    distances = torch.linalg.vector_norm(target_token - target_token_gt[:, None, :], dim=-1)
    candidate_target = torch.zeros(128, 9, dtype=torch.float32)
    recoverable = in_bounds & (distances.min(dim=-1).values <= 2.0)
    soft = torch.exp(-distances.square() / (2.0 * 0.5**2))
    soft = soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    candidate_target[recoverable, :8] = soft[recoverable]
    candidate_target[~recoverable, 8] = 1.0

    axis = torch.linspace(0.0, 511.0, 16)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    grid_flow = _sample_truth_flow(flow, grid_xy)
    return {
        "candidate_target": candidate_target,
        "target_token_gt": target_token_gt,
        "candidate_distance_gt": distances,
        "recoverable": recoverable,
        "grid_xy_pixel": grid_xy,
        "grid_truth_flow": grid_flow,
        "uses_raw_truth_flow": True,
        "uses_datum": False,
    }


def _atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _candidate_record(record, table, checkpoint_sha, anchor_set):
    return {
        **table,
        "pair_id": record["pair_id"],
        "split": record["split"],
        "checkpoint_sha256": checkpoint_sha,
        "anchor_sha256": anchor_set.sha256,
        "candidate_count": 8,
        "anchor_count": 128,
        "score_type": "cosine",
        "image_size": [512, 512],
        "feature_size": [64, 64],
        "uses_ground_truth": False,
        "uses_datum": False,
        "optical_sha256": record["files"]["opt"]["sha256"],
        "sar_sha256": record["files"]["sar_warped"]["sha256"],
    }


def build_cache(args):
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("GDROS checkpoint SHA mismatch")
    records = load_split_manifest(args.split_manifest)
    anchor_set = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))
    if anchor_set.sha256 != ANCHOR_SHA256:
        raise RuntimeError(f"anchor SHA mismatch: {anchor_set.sha256}")
    device = torch.device(args.device)
    model, _, _ = load_slim_feature(args.checkpoint, device)
    model.requires_grad_(False).eval()
    selected = [
        row for index, row in enumerate(records) if index % args.shard_count == args.shard_index
    ]
    if args.split_filter is not None:
        selected = [row for row in selected if row["split"] == args.split_filter]
    if args.max_pairs is not None:
        selected = selected[: args.max_pairs]
    completed = []
    for position, record in enumerate(selected):
        enforce_split_access(record, args.training_root, record["split"])
        candidate_path = args.cache_root / "candidates" / record["split"] / f"{record['pair_id']}.pt"
        label_path = args.cache_root / "labels" / record["split"] / f"{record['pair_id']}.pt"
        if candidate_path.exists() and label_path.exists() and not args.overwrite:
            completed.append(record["pair_id"])
            continue
        optical = _load_image(Path(record["files"]["opt"]["path"])).to(device)
        sar = _load_image(Path(record["files"]["sar_warped"]["path"])).to(device)
        table = build_frozen_candidate_table(model, optical, sar, anchor_set)
        candidate_record = _candidate_record(record, table, CHECKPOINT_SHA256, anchor_set)
        flow = read_flow(Path(record["files"]["truth_flow"]["path"]))
        labels = build_training_labels(candidate_record, flow)
        label_record = {
            **labels,
            "pair_id": record["pair_id"],
            "split": record["split"],
            "truth_flow_sha256": record["files"]["truth_flow"]["sha256"],
        }
        _atomic_torch_save(candidate_record, candidate_path)
        _atomic_torch_save(label_record, label_path)
        completed.append(record["pair_id"])
        if position % args.log_every == 0:
            print(f"shard {args.shard_index}: {position}/{len(selected)} {record['pair_id']}", flush=True)
    result = {
        "status": "PASS",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pair_count": len(completed),
        "pair_ids": completed,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "anchor_sha256": anchor_set.sha256,
        "uses_ground_truth_in_candidate_cache": False,
        "labels_use_raw_truth_flow": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "shard_index", "pair_count")}, sort_keys=True))


def verify_cache_recompute_parity(args):
    records = load_split_manifest(args.split_manifest)[:2]
    anchor_set = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))
    device = torch.device(args.device)
    model, _, _ = load_slim_feature(args.checkpoint, device)
    model.requires_grad_(False).eval()
    results = []
    for record in records:
        cached = torch.load(
            args.cache_root / "candidates" / record["split"] / f"{record['pair_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        online = build_frozen_candidate_table(
            model,
            _load_image(Path(record["files"]["opt"]["path"])).to(device),
            _load_image(Path(record["files"]["sar_warped"]["path"])).to(device),
            anchor_set,
        )
        keys = (
            "candidate_features",
            "candidate_scores",
            "candidate_indices",
            "source_xy_norm",
            "target_xy_norm",
        )
        maximum = {}
        for key in keys:
            if cached[key].dtype.is_floating_point:
                maximum[key] = float((cached[key] - online[key].cpu()).abs().max().item())
                if maximum[key] != 0.0:
                    raise RuntimeError(f"cache differential failed for {record['pair_id']} {key}")
            elif not torch.equal(cached[key], online[key].cpu()):
                raise RuntimeError(f"cache differential failed for {record['pair_id']} {key}")
        results.append({"pair_id": record["pair_id"], "max_abs": maximum})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "pairs": results}, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--split-filter", choices=("train", "dev"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs

    validate_frozen_specs()
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.command == "build":
        build_cache(args)
    else:
        verify_cache_recompute_parity(args)


if __name__ == "__main__":
    main()
