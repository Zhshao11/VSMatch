#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sparsegeo.e003a.candidate_cache import read_flow
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest, sha256_file
from experiments.sparsegeo.e003a.evaluate_selector import _pair_metrics, _summarize
from experiments.sparsegeo.e003a.selector import weighted_affine_solve


@torch.no_grad()
def run(args):
    records = load_split_manifest(args.split_manifest, "dev")
    device = torch.device(args.device)
    errors = []
    ready = []
    grids = []
    valid_count = 0
    recoverable_counts = []
    rank_counts = np.zeros(8, dtype=np.int64)
    rows = []
    for record in records:
        pair_id = record["pair_id"]
        candidate = torch.load(
            args.cache_root / "candidates/dev" / f"{pair_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        labels = torch.load(
            args.cache_root / "labels/dev" / f"{pair_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        recoverable = labels["recoverable"].bool().to(device)
        best = labels["candidate_distance_gt"].argmin(dim=-1).to(device)
        weights = torch.zeros(1, 128, 8, device=device)
        weights.scatter_(2, best.view(1, 128, 1), recoverable.view(1, 128, 1).float())
        affine, condition, valid = weighted_affine_solve(
            candidate["source_xy_norm"].unsqueeze(0).to(device),
            candidate["target_xy_norm"].unsqueeze(0).to(device),
            weights,
        )
        pair_epe, geometry = _pair_metrics(
            affine,
            read_flow(Path(record["files"]["truth_flow"]["path"])),
            labels["grid_xy_pixel"].to(device),
            labels["grid_truth_flow"].to(device),
        )
        count = int(recoverable.sum().item())
        recoverable_counts.append(count)
        if count:
            rank_counts += np.bincount(best[recoverable].cpu().numpy(), minlength=8)
        valid_count += int(valid.item())
        errors.append(pair_epe)
        ready.append(geometry["pair_local_ready"])
        grids.append(geometry["grid_mean_l2_px"])
        rows.append(
            {
                "pair_id": pair_id,
                "pair_epe": pair_epe,
                "pair_local_ready": geometry["pair_local_ready"],
                "grid_mean_l2_px": geometry["grid_mean_l2_px"],
                "recoverable_anchor_count": count,
                "solver_valid": bool(valid.item()),
                "condition": float(condition.item()),
            }
        )
    result = {
        "experiment": "E-003A",
        "audit": "corrected-top8-gt-oracle",
        "status": "PASS",
        "pair_count": len(rows),
        "gt_access": "post-formal-method-output-diagnostic-only",
        "datum_used": False,
        "oracle_rule": "nearest top8 candidate only for frozen recoverable=true anchors; zero weight otherwise",
        "metrics": {
            **_summarize(errors),
            "pair_local_ready_percent": float(100.0 * np.mean(ready)),
            "grid_mean_l2_px_mean": float(np.mean(grids)),
        },
        "solver_valid_percent": float(100.0 * valid_count / len(rows)),
        "recoverable_anchor_percent": float(100.0 * np.sum(recoverable_counts) / (128 * len(rows))),
        "recoverable_anchor_count_p50": float(np.median(recoverable_counts)),
        "recoverable_best_rank_0based_counts": rank_counts.tolist(),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "per_pair": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "metrics", "solver_valid_percent", "recoverable_anchor_percent")}, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
