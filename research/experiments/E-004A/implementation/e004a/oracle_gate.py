#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sparsegeo.e003a.candidate_cache import read_flow
from experiments.sparsegeo.e003a.data_protocol import enforce_split_access, load_split_manifest, sha256_file
from experiments.sparsegeo.e004a.evaluator import pair_metrics, summarize
from experiments.sparsegeo.e004a.proposal_cache import _paths, validate_cached_pair
from experiments.sparsegeo.e004a.proposals import (
    SELECTED_COUNT, allocate_oracle_k128, build_region_proposals, region_counts, uniform_region_k128,
)
from experiments.sparsegeo.e004a.provenance import runtime_provenance


MIN_CORRESPONDENCES = 12
MIN_QUADRANTS = 3
MAX_CONDITION = 20.0


def _quadrant_count(source_pixel):
    x_right = source_pixel[:, 0] >= 255.5
    y_bottom = source_pixel[:, 1] >= 255.5
    return int(torch.unique(x_right.to(torch.long) + 2 * y_bottom.to(torch.long)).numel())


def fit_selected_oracle(candidate_table, labels, selected):
    selected = selected.to(torch.long).reshape(-1).cpu()
    if selected.numel() != SELECTED_COUNT:
        raise ValueError("selected-anchor oracle requires strict K128")
    recoverable = labels["recoverable"].to(torch.bool).index_select(0, selected)
    usable = selected[recoverable]
    result = {
        "selected_count": int(selected.numel()),
        "recoverable_count": int(usable.numel()),
        "recoverable_fraction": float(usable.numel() / selected.numel()),
        "correspondence_count": int(usable.numel()),
        "quadrant_count": 0,
        "condition": None,
        "valid": False,
        "invalid_reason": None,
        "affine": None,
    }
    if usable.numel() < MIN_CORRESPONDENCES:
        result["invalid_reason"] = "fewer_than_12_correspondences"
        return result
    source_pixel = candidate_table["source_xy_pixel"].to(torch.float32).index_select(0, usable)
    slots = labels["nearest_candidate_slot"].to(torch.long).index_select(0, usable)
    targets_all = candidate_table["target_xy_token"].to(torch.float32).index_select(0, usable) * 8.0
    target_pixel = targets_all[torch.arange(usable.numel()), slots]
    if not bool(torch.isfinite(source_pixel).all() and torch.isfinite(target_pixel).all()):
        result["invalid_reason"] = "nonfinite_correspondence"
        return result
    result["quadrant_count"] = _quadrant_count(source_pixel)
    if result["quadrant_count"] < MIN_QUADRANTS:
        result["invalid_reason"] = "fewer_than_3_quadrants"
        return result
    source_normalized = 2.0 * source_pixel / 511.0 - 1.0
    target_normalized = 2.0 * target_pixel / 511.0 - 1.0
    design = torch.cat((source_normalized, torch.ones_like(source_normalized[:, :1])), dim=-1)
    condition = torch.linalg.cond(design.transpose(0, 1) @ design)
    result["condition"] = float(condition.item()) if torch.isfinite(condition) else None
    if not bool(torch.isfinite(condition)) or float(condition) > MAX_CONDITION:
        result["invalid_reason"] = "condition_exceeds_20"
        return result
    solution = torch.linalg.lstsq(design.to(torch.float32), target_normalized.to(torch.float32)).solution
    affine = solution.transpose(0, 1).contiguous().unsqueeze(0)
    if affine.shape != (1, 2, 3) or not bool(torch.isfinite(affine).all()):
        result["invalid_reason"] = "nonfinite_affine"
        return result
    result.update({"valid": True, "invalid_reason": None, "affine": affine})
    return result


def evaluate_anchor_set(candidate_table, labels, selected, flow_truth, grid_xy, grid_truth):
    fit = fit_selected_oracle(candidate_table, labels, selected)
    row = {key: value for key, value in fit.items() if key != "affine"}
    if not fit["valid"]:
        return {**row, "aepe": None, "pair_local_ready": False, "geometry": None}
    aepe, geometry = pair_metrics(fit["affine"], flow_truth, grid_xy, grid_truth)
    return {**row, "aepe": aepe, "pair_local_ready": geometry["pair_local_ready"], "geometry": geometry}


def _variant_summary(rows, name):
    values = [row["variants"][name] for row in rows]
    valid = [value for value in values if value["valid"]]
    conditions = [value["condition"] for value in values if value["condition"] is not None]
    aepe = [value["aepe"] for value in valid]
    reasons = sorted({v["invalid_reason"] for v in values if v["invalid_reason"]})
    return {
        "pair_count": len(values),
        "mean_recoverable_count": float(np.mean([v["recoverable_count"] for v in values])),
        "median_recoverable_count": float(np.median([v["recoverable_count"] for v in values])),
        "recoverable_fraction": float(np.mean([v["recoverable_fraction"] for v in values])),
        "valid_pair_percent": float(100.0 * np.mean([v["valid"] for v in values])),
        "pair_local_ready_percent": float(100.0 * np.mean([v["pair_local_ready"] for v in values])),
        "condition_max": max(conditions, default=None),
        "metrics": summarize(aepe) if aepe else None,
        "invalid_reason_counts": {reason: sum(v["invalid_reason"] == reason for v in values) for reason in reasons},
    }


def run_p0(args):
    provenance = runtime_provenance(
        args.spec_root, args.checkpoint, args.split_manifest, Path(__file__).resolve().parent,
    )
    records = load_split_manifest(args.split_manifest, allowed_split="dev")
    proposals = build_region_proposals()
    uniform = uniform_region_k128(proposals)
    rows = []
    cache_files = []
    for position, record in enumerate(records):
        enforce_split_access(record, args.training_root, "dev")
        candidate_path, label_path = _paths(args.cache_root, "dev", record["pair_id"])
        if not candidate_path.exists() or not label_path.exists():
            raise RuntimeError(f"incomplete P0 cache for dev pair {record['pair_id']}")
        candidates = torch.load(candidate_path, map_location="cpu", weights_only=True)
        labels = torch.load(label_path, map_location="cpu", weights_only=True)
        validate_cached_pair(
            candidates, labels, record, "dev", provenance["source"]["sha256"],
            provenance["checkpoint_sha256"],
        )
        cache_files.append({
            "pair_id": record["pair_id"],
            "candidate_sha256": sha256_file(candidate_path),
            "label_sha256": sha256_file(label_path),
        })
        proposal_labels = labels["proposal_m256"]
        oracle = allocate_oracle_k128(
            proposal_labels["recoverable"], proposal_labels["nearest_candidate_error"], proposals,
        )
        counts = region_counts(oracle, proposals)
        if counts.min().item() < 1 or counts.max().item() > 3:
            raise RuntimeError("oracle allocation violates region bounds")
        # The raw flow is opened only after cached non-GT candidates and allocation are validated.
        flow_truth = read_flow(Path(record["files"]["truth_flow"]["path"]))
        grid_xy = labels["grid_xy_pixel"].to(torch.float32)
        grid_truth = labels["grid_truth_flow"].to(torch.float32)
        variants = {
            "fixed-public-K128": evaluate_anchor_set(
                candidates["fixed_public_k128"], labels["fixed_public_k128"], torch.arange(128),
                flow_truth, grid_xy, grid_truth,
            ),
            "uniform-region-K128": evaluate_anchor_set(
                candidates["proposal_m256"], proposal_labels, uniform, flow_truth, grid_xy, grid_truth,
            ),
            "gt-constrained-proposal-K128": evaluate_anchor_set(
                candidates["proposal_m256"], proposal_labels, oracle, flow_truth, grid_xy, grid_truth,
            ),
        }
        rows.append({
            "pair_id": record["pair_id"], "variants": variants,
            "oracle_selected_proposal_ids": oracle.tolist(),
            "oracle_region_counts": counts.tolist(),
            "oracle_all_regions_covered": bool((counts >= 1).all()),
        })
        if position % args.log_every == 0:
            print(f"P0 evaluator: {position}/{len(records)} {record['pair_id']}", flush=True)
    names = ("fixed-public-K128", "uniform-region-K128", "gt-constrained-proposal-K128")
    summaries = {name: _variant_summary(rows, name) for name in names}
    main, fixed, uniform_summary = (
        summaries["gt-constrained-proposal-K128"], summaries["fixed-public-K128"],
        summaries["uniform-region-K128"],
    )
    gates = {
        "mean_recoverable_gain_vs_fixed_ge20": main["mean_recoverable_count"] - fixed["mean_recoverable_count"] >= 20.0,
        "mean_recoverable_gain_vs_uniform_ge10": main["mean_recoverable_count"] - uniform_summary["mean_recoverable_count"] >= 10.0,
        "selected_recoverable_fraction_ge50pct": main["recoverable_fraction"] >= 0.50,
        "median_recoverable_count_ge64": main["median_recoverable_count"] >= 64.0,
        "oracle_aepe_le2_5": main["metrics"] is not None and main["metrics"]["aepe"] <= 2.5,
        "pair_local_ready_ge99pct": main["pair_local_ready_percent"] >= 99.0,
        "oracle_valid_pairs_ge99pct": main["valid_pair_percent"] >= 99.0,
        "all_regions_covered": all(row["oracle_all_regions_covered"] for row in rows),
        "source_design_condition_le20": main["condition_max"] is not None and main["condition_max"] <= 20.0,
    }
    cache_encoded = json.dumps(cache_files, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stage": "P0_PROPOSAL_ORACLE_HEADROOM", "pair_count": len(rows),
        "provenance": {
            **provenance,
            "cache_manifest": {
                "pair_count": len(cache_files),
                "sha256": hashlib.sha256(cache_encoded).hexdigest(),
                "files": cache_files,
            },
        }, "summaries": summaries,
        "deltas": {
            "mean_recoverable_vs_fixed": main["mean_recoverable_count"] - fixed["mean_recoverable_count"],
            "mean_recoverable_vs_uniform": main["mean_recoverable_count"] - uniform_summary["mean_recoverable_count"],
        },
        "gates": gates, "gt_nearest_top8_oracle_only": True, "full_image_no_mask": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.per_pair_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.per_pair_output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "gates": gates}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--spec-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-pair-output", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_p0(parse_args())
