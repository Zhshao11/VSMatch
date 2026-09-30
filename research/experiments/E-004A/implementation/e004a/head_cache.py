#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from experiments.sparsegeo.e002a.feature import load_slim_feature
from experiments.sparsegeo.e003a.candidate_cache import _load_image, read_flow
from experiments.sparsegeo.e003a.data_protocol import enforce_split_access, load_split_manifest
from experiments.sparsegeo.e004a.coarse_probe import compute_coarse_probe_features
from experiments.sparsegeo.e004a.proposal_cache import _atomic_torch_save, build_table_labels, exact_cosine_top8
from experiments.sparsegeo.e004a.proposals import build_region_proposals
from experiments.sparsegeo.e004a.provenance import runtime_provenance


def validate_head_cache(candidate,label,record,split,source_sha,checkpoint_sha):
    for key,value in (("pair_id",record["pair_id"]),("split",split),("source_manifest_sha256",source_sha)):
        if candidate.get(key)!=value or label.get(key)!=value: raise RuntimeError(f"head cache {key} mismatch")
    checks=(candidate.get("checkpoint_sha256")==checkpoint_sha,candidate.get("optical_sha256")==record["files"]["opt"]["sha256"],candidate.get("sar_sha256")==record["files"]["sar_warped"]["sha256"],label.get("truth_flow_sha256")==record["files"]["truth_flow"]["sha256"],candidate.get("uses_ground_truth") is False,label.get("uses_datum") is False)
    if not all(checks): raise RuntimeError(f"head cache provenance mismatch {record['pair_id']}")
    if candidate["pre_features"].shape!=(256,11) or candidate["candidate_scores"].shape!=(256,8) or label["recoverable"].shape!=(256,): raise RuntimeError("head cache schema mismatch")
    return True


@torch.inference_mode()
def build_head_pair(model, optical, sar):
    source, target = model(optical, sar)
    proposals = build_region_proposals()
    scores, indices, xy = exact_cosine_top8(source, target, proposals.lattice_indices)
    pre_features, coarse_xy, coarse_scores = compute_coarse_probe_features(source, target)
    return {
        "source_xy_pixel": proposals.pixel_xy,
        "source_lattice_indices": proposals.lattice_indices,
        "proposal_id": proposals.proposal_id,
        "region_id": proposals.region_id,
        "pre_features": pre_features.cpu(),
        "coarse_top1_xy": coarse_xy.cpu(),
        "candidate_scores": scores,
        "candidate_indices": indices,
        "target_xy_token": xy,
        "coarse_score_shape": list(coarse_scores.shape),
    }


def run(args):
    provenance = runtime_provenance(args.spec_root, args.checkpoint, args.split_manifest, Path(__file__).parent)
    records = load_split_manifest(args.split_manifest, allowed_split=args.split)
    records = [r for i, r in enumerate(records) if i % args.shard_count == args.shard_index]
    device = torch.device(args.device)
    model, _, _ = load_slim_feature(args.checkpoint, device); model.requires_grad_(False).eval()
    completed = []
    for position, record in enumerate(records):
        enforce_split_access(record, args.training_root, args.split)
        cp = args.cache_root / "candidates" / args.split / f"{record['pair_id']}.pt"
        lp = args.cache_root / "labels" / args.split / f"{record['pair_id']}.pt"
        if cp.exists() and lp.exists():
            validate_head_cache(torch.load(cp,map_location="cpu",weights_only=True),torch.load(lp,map_location="cpu",weights_only=True),record,args.split,provenance["source"]["sha256"],provenance["checkpoint_sha256"])
            completed.append(record["pair_id"]); continue
        if cp.exists() or lp.exists():
            raise RuntimeError(f"partial head cache {record['pair_id']}")
        optical = _load_image(Path(record["files"]["opt"]["path"])).to(device)
        sar = _load_image(Path(record["files"]["sar_warped"]["path"])).to(device)
        table = build_head_pair(model, optical, sar)
        labels = build_table_labels(table, read_flow(Path(record["files"]["truth_flow"]["path"])))
        _atomic_torch_save({
            **table, "pair_id": record["pair_id"], "split": args.split,
            "source_manifest_sha256": provenance["source"]["sha256"], "uses_ground_truth": False,
            "checkpoint_sha256":provenance["checkpoint_sha256"],"optical_sha256":record["files"]["opt"]["sha256"],"sar_sha256":record["files"]["sar_warped"]["sha256"],
        }, cp)
        _atomic_torch_save({
            **labels, "pair_id": record["pair_id"], "split": args.split,
            "truth_flow_sha256": record["files"]["truth_flow"]["sha256"], "uses_datum": False,
            "source_manifest_sha256":provenance["source"]["sha256"],
        }, lp)
        completed.append(record["pair_id"])
        if position % 10 == 0: print(f"head-cache {args.split} {args.shard_index}: {position}/{len(records)}", flush=True)
    result = {"status":"PASS","split":args.split,"shard":args.shard_index,"count":len(completed),"provenance":provenance}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


def parse_args():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","split-manifest","training-root","spec-root","cache-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--split",choices=("train","dev"),required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--shard-count",type=int,default=1); p.add_argument("--shard-index",type=int,default=0)
    return p.parse_args()


if __name__ == "__main__": run(parse_args())
