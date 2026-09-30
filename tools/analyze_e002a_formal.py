#!/usr/bin/env python3
"""Independent, artifact-only audit and proxy analysis for E-002A formal runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIR_COUNT = 424
BOOTSTRAP_REPLICATES = 2000
VARIANT_IDS_FOR_GROUPS = ("native_raw_hard_k128", "native_cosine_hard_k128")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def as_float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest = load_json(root / "artifact_manifest.json")
    issues = []
    checked = 0
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            issues.append(f"missing:{relative}")
            continue
        checked += 1
        size = path.stat().st_size
        digest = sha256(path)
        if size != expected["size_bytes"]:
            issues.append(f"size:{relative}:{size}!={expected['size_bytes']}")
        if digest != expected["sha256"]:
            issues.append(f"sha256:{relative}:{digest}!={expected['sha256']}")
    return {"passed": not issues, "checked_files": checked, "issues": issues}


def close(left: float, right: float, atol: float = 1e-9) -> bool:
    return bool(abs(left - right) <= atol)


def pair_bootstrap(values: np.ndarray, indices: np.ndarray, statistic: str) -> dict:
    sampled = values[indices]
    if statistic == "mean":
        estimates = np.nanmean(sampled, axis=1)
        point = float(np.nanmean(values))
    elif statistic == "p95":
        estimates = np.nanpercentile(sampled, 95, axis=1)
        point = float(np.nanpercentile(values, 95))
    else:
        raise ValueError(statistic)
    return {
        "point": point,
        "p2_5": float(np.nanpercentile(estimates, 2.5)),
        "p97_5": float(np.nanpercentile(estimates, 97.5)),
        "replicates": int(indices.shape[0]),
        "unit": "image-pair",
    }


def independent_group_bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = np.mean(values[draws], axis=1)
    return {
        "point": float(np.mean(values)),
        "p2_5": float(np.percentile(estimates, 2.5)),
        "p97_5": float(np.percentile(estimates, 97.5)),
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "image-pair",
    }


def audit_accuracy(root: Path) -> tuple[dict, dict, dict]:
    metrics = load_json(root / "metrics.json")
    gates = load_json(root / "gates.json")
    per_pair = read_csv(root / "per_pair.csv")
    retrieval_pair = read_csv(root / "retrieval_per_pair.csv")
    retrieval_anchor = read_csv(root / "retrieval_per_anchor.csv")
    oracle_rows = read_csv(root / "oracles_per_pair.csv")
    condition_rows = read_csv(root / "conditions.csv")
    bootstrap_indices = np.load(root / "bootstrap_pair_indices.npy")
    with np.load(root / "retrieval_trace.npz") as trace_file:
        trace = {key: trace_file[key] for key in trace_file.files}

    issues = []
    variant_ids = sorted(metrics["variant_results"])
    expected_per_pair = PAIR_COUNT * len(variant_ids)
    if len(per_pair) != expected_per_pair:
        issues.append(f"per_pair_count:{len(per_pair)}!={expected_per_pair}")
    pair_variant_keys = {(int(row["pair_index"]), row["variant_id"]) for row in per_pair}
    if len(pair_variant_keys) != expected_per_pair:
        issues.append("per_pair_duplicate_or_missing_keys")
    if {int(row["pair_index"]) for row in per_pair} != set(range(PAIR_COUNT)):
        issues.append("per_pair_pair_index_domain")
    if any(as_bool(row["uses_ground_truth"]) for row in per_pair):
        issues.append("non_gt_variant_marked_as_gt")

    by_variant = defaultdict(list)
    for row in per_pair:
        by_variant[row["variant_id"]].append(row)
    recomputed_variants = {}
    bootstrap_ci = {"variants": {}, "retrieval_non_recall": {}, "oracles": {}}
    max_metric_abs_diff = 0.0
    for variant_id in variant_ids:
        rows = sorted(by_variant[variant_id], key=lambda row: int(row["pair_index"]))
        if len(rows) != PAIR_COUNT:
            issues.append(f"variant_pair_count:{variant_id}:{len(rows)}")
            continue
        errors = np.asarray([as_float(row["pair_public_error"]) for row in rows], dtype=np.float64)
        ready = np.asarray([as_bool(row["pair_local_ready"]) for row in rows], dtype=np.float64)
        grid_mean = np.asarray([as_float(row["grid_mean_l2_px"]) for row in rows], dtype=np.float64)
        capture = np.asarray([as_float(row["capture_at_32_percent"]) for row in rows], dtype=np.float64)
        if not np.isfinite(errors).all() or not np.isfinite(grid_mean).all():
            issues.append(f"nonfinite_variant:{variant_id}")
        public = metrics["variant_results"][variant_id]["public_metric"]
        summary = {
            "aepe": float(errors.mean()),
            "cmr1_percent": float((errors < 1).mean() * 100),
            "cmr2_percent": float((errors < 2).mean() * 100),
            "cmr5_percent": float((errors < 5).mean() * 100),
            "failure_percent": float((errors >= 5).mean() * 100),
            "pair_error_p50": float(np.percentile(errors, 50)),
            "pair_error_p95": float(np.percentile(errors, 95)),
            "pair_error_p99": float(np.percentile(errors, 99)),
            "pair_local_ready_percent": float(ready.mean() * 100),
            "grid_mean_l2_px_mean": float(grid_mean.mean()),
            "capture_at_32_percent_mean": float(capture.mean()),
        }
        comparisons = [
            (summary["aepe"], public["aepe"]),
            (summary["cmr1_percent"], public["thresholds"]["1"]["cmr_percent"]),
            (summary["cmr2_percent"], public["thresholds"]["2"]["cmr_percent"]),
            (summary["cmr5_percent"], public["thresholds"]["5"]["cmr_percent"]),
            (summary["failure_percent"], public["failure_rate_percent"]),
        ]
        max_metric_abs_diff = max(max_metric_abs_diff, *(abs(a - b) for a, b in comparisons))
        recomputed_variants[variant_id] = summary
        bootstrap_ci["variants"][variant_id] = {
            "aepe": pair_bootstrap(errors, bootstrap_indices, "mean"),
            "pair_error_p95": pair_bootstrap(errors, bootstrap_indices, "p95"),
            "cmr1_percent": pair_bootstrap((errors < 1).astype(np.float64) * 100, bootstrap_indices, "mean"),
            "cmr2_percent": pair_bootstrap((errors < 2).astype(np.float64) * 100, bootstrap_indices, "mean"),
            "cmr5_percent": pair_bootstrap((errors < 5).astype(np.float64) * 100, bootstrap_indices, "mean"),
            "pair_local_ready_percent": pair_bootstrap(ready * 100, bootstrap_indices, "mean"),
            "grid_mean_l2_px": pair_bootstrap(grid_mean, bootstrap_indices, "mean"),
        }
    if max_metric_abs_diff > 1e-9:
        issues.append(f"variant_json_csv_max_diff:{max_metric_abs_diff}")

    condition_rank_ok = all(int(row["rank"]) == 3 for row in condition_rows)
    condition_max = max(as_float(row["condition"]) for row in condition_rows)
    if len(condition_rows) != 6 or not condition_rank_ok or condition_max > 10:
        issues.append("anchor_condition_contract")

    retrieval_keys = {(int(row["pair_index"]), row["score_type"]) for row in retrieval_pair}
    if len(retrieval_pair) != PAIR_COUNT * 2 or len(retrieval_keys) != PAIR_COUNT * 2:
        issues.append("retrieval_per_pair_contract")
    retrieval_recomputed = {}
    retrieval_rows_by_score = {}
    for score_type in ("raw", "cosine"):
        rows = sorted(
            [row for row in retrieval_pair if row["score_type"] == score_type],
            key=lambda row: int(row["pair_index"]),
        )
        retrieval_rows_by_score[score_type] = rows
        inbounds = np.asarray([int(row["inbounds_count"]) for row in rows], dtype=np.float64)
        top1_ready = np.asarray([
            int(row["top1_inlier_count"]) >= 12
            and int(row["top1_inlier_quadrants"]) >= 3
            and as_float(row["top1_inlier_condition"]) <= 20
            for row in rows
        ], dtype=np.float64)
        top32_ready = np.asarray([
            int(row["recoverable32_count"]) >= 16
            and int(row["recoverable32_quadrants"]) >= 3
            and as_float(row["recoverable32_condition"]) <= 20
            for row in rows
        ], dtype=np.float64)
        top64_bad = np.asarray([
            int(row["recoverable64_r2_count"]) < 6 or int(row["recoverable64_r2_quadrants"]) < 3
            for row in rows
        ], dtype=np.float64)
        retrieval_recomputed[score_type] = {
            "inbounds_anchor_percent": float(inbounds.sum() / (PAIR_COUNT * 128) * 100),
            "pair_top1_structural_ready_percent": float(top1_ready.mean() * 100),
            "pair_top32_recoverable_structural_percent": float(top32_ready.mean() * 100),
            "pair_top64_bad_percent": float(top64_bad.mean() * 100),
            "recall_micro": {},
        }
        for k in (1, 2, 4, 8, 16, 32, 64):
            for radius in (1, 2, 4):
                column = f"recall_num_k{k}_r{radius}"
                numerator = np.asarray([int(row[column]) for row in rows], dtype=np.float64)
                value = float(numerator.sum() / inbounds.sum())
                retrieval_recomputed[score_type]["recall_micro"][f"R@{k},r={radius}"] = value
                recorded = metrics["retrieval"][score_type]["recall"][f"R@{k},r={radius}"]["micro"]
                if not close(value, recorded):
                    issues.append(f"retrieval_recall:{score_type}:k{k}:r{radius}")

        for name, column in (
            ("inbounds_anchor_percent", None),
            ("top1_structural_ready_percent", None),
            ("top32_recoverable_structural_percent", None),
            ("top64_bad_percent", None),
            ("unique_target_ratio_mean", "unique_target_ratio"),
            ("margin_top32_precision_mean", "margin_top32_precision_all"),
            ("entropy_top32_precision_mean", "entropy_top32_precision_all"),
        ):
            if column is None:
                if name == "inbounds_anchor_percent":
                    values = inbounds / 128 * 100
                elif name == "top1_structural_ready_percent":
                    values = top1_ready * 100
                elif name == "top32_recoverable_structural_percent":
                    values = top32_ready * 100
                else:
                    values = top64_bad * 100
            else:
                values = np.asarray([as_float(row[column]) for row in rows], dtype=np.float64)
            bootstrap_ci["retrieval_non_recall"][f"{score_type}:{name}"] = pair_bootstrap(
                values, bootstrap_indices, "mean"
            )

    if len(oracle_rows) != PAIR_COUNT * 33:
        issues.append(f"oracle_row_count:{len(oracle_rows)}")
    if len({(int(row["pair_index"]), row["oracle_id"]) for row in oracle_rows}) != len(oracle_rows):
        issues.append("oracle_duplicate_keys")
    if not all(as_bool(row["uses_ground_truth"]) for row in oracle_rows):
        issues.append("oracle_gt_label")
    oracle_by_id = defaultdict(list)
    for row in oracle_rows:
        oracle_by_id[row["oracle_id"]].append(row)
    oracle_recomputed = {}
    for oracle_id in (
        "continuous_gt_k128",
        "nearest_lattice_inbounds_k128",
        "topk_raw_k32_select_and_reject",
        "topk_cosine_k32_select_and_reject",
        "topk_raw_k64_select_and_reject",
        "topk_cosine_k64_select_and_reject",
    ):
        rows = sorted(oracle_by_id[oracle_id], key=lambda row: int(row["pair_index"]))
        available = np.asarray([as_bool(row["available"]) for row in rows])
        ready = np.asarray([as_bool(row["pair_local_ready"]) for row in rows], dtype=np.float64)
        grid = np.asarray([as_float(row["grid_mean_l2_px"]) for row in rows], dtype=np.float64)
        valid_grid = grid[available]
        oracle_recomputed[oracle_id] = {
            "available_percent": float(available.mean() * 100),
            "pair_local_ready_percent": float(ready[available].mean() * 100),
            "grid_mean_l2_px": float(np.nanmean(valid_grid)),
        }
        bootstrap_ci["oracles"][f"{oracle_id}:pair_local_ready_percent"] = pair_bootstrap(
            ready * 100, bootstrap_indices, "mean"
        )

    expected_anchor_rows = PAIR_COUNT * 2 * 128
    anchor_keys = {
        (int(row["pair_index"]), row["score_type"], int(row["anchor_position"]))
        for row in retrieval_anchor
    }
    if len(retrieval_anchor) != expected_anchor_rows or len(anchor_keys) != expected_anchor_rows:
        issues.append("retrieval_per_anchor_contract")
    trace_shapes = {key: list(value.shape) for key, value in trace.items()}
    expected_trace_shape = [PAIR_COUNT, 128, 64]
    for key in ("raw_top_indices", "raw_top_scores", "cosine_top_indices", "cosine_top_scores"):
        if trace_shapes.get(key) != expected_trace_shape:
            issues.append(f"trace_shape:{key}:{trace_shapes.get(key)}")
    for score_type in ("raw", "cosine"):
        indices = trace[f"{score_type}_top_indices"]
        scores = trace[f"{score_type}_top_scores"]
        if indices.min() < 0 or indices.max() >= 4096:
            issues.append(f"trace_index_range:{score_type}")
        if not np.isfinite(scores).all():
            issues.append(f"trace_nonfinite:{score_type}")
        if np.any(scores[..., :-1] < scores[..., 1:]):
            issues.append(f"trace_score_order:{score_type}")
    top1_mismatch = 0
    anchor_aggregates = defaultdict(lambda: {"entropy": [], "margin": []})
    for row in retrieval_anchor:
        pair = int(row["pair_index"])
        position = int(row["anchor_position"])
        score_type = row["score_type"]
        if int(row["top1_target_index"]) != int(trace[f"{score_type}_top_indices"][pair, position, 0]):
            top1_mismatch += 1
        anchor_aggregates[(pair, score_type)]["entropy"].append(as_float(row["entropy"]))
        anchor_aggregates[(pair, score_type)]["margin"].append(as_float(row["margin"]))
    if top1_mismatch:
        issues.append(f"trace_top1_mismatch:{top1_mismatch}")

    if bootstrap_indices.shape != (BOOTSTRAP_REPLICATES, PAIR_COUNT):
        issues.append(f"bootstrap_shape:{bootstrap_indices.shape}")
    if bootstrap_indices.min() < 0 or bootstrap_indices.max() >= PAIR_COUNT:
        issues.append("bootstrap_index_domain")

    audit = {
        "passed": not issues,
        "issues": issues,
        "artifact_manifest": verify_manifest(root),
        "formal_contract": {
            "formal_run": metrics["formal_run"],
            "formal_gate_applicable": metrics["formal_gate_applicable"],
            "pairs": metrics["num_evaluated_pairs"],
            "provenance_passed": metrics["provenance_consistency"]["passed"],
            "gt_isolation_passed": metrics["gt_isolation"]["passed"],
            "finite_validation_passed": metrics["finite_validation"]["passed"],
        },
        "row_counts": {
            "per_pair": len(per_pair),
            "retrieval_per_pair": len(retrieval_pair),
            "retrieval_per_anchor": len(retrieval_anchor),
            "oracles_per_pair": len(oracle_rows),
            "conditions": len(condition_rows),
        },
        "variant_json_csv_max_abs_diff": max_metric_abs_diff,
        "conditions": {"all_rank3": condition_rank_ok, "max_condition": condition_max},
        "trace": {"shapes": trace_shapes, "top1_mismatch_count": top1_mismatch},
        "gates_G0_G4": gates,
        "variants": recomputed_variants,
        "retrieval": retrieval_recomputed,
        "oracles": oracle_recomputed,
    }
    return audit, bootstrap_ci, {
        "by_variant": by_variant,
        "retrieval_rows_by_score": retrieval_rows_by_score,
        "anchor_aggregates": anchor_aggregates,
        "bootstrap_indices": bootstrap_indices,
    }


def transform_proxy(dataset_root: Path, pair_index: int) -> float:
    datum = np.load(
        dataset_root / "testing" / "datum" / f"{pair_index:06d}.npy", allow_pickle=True
    )
    affine = np.asarray(datum[2], dtype=np.float64)
    points = np.asarray([[0, 0], [511, 0], [0, 511], [511, 511], [255.5, 255.5]], dtype=np.float64)
    predicted = points @ affine[:, :2].T + affine[:, 2]
    displacement = predicted - points
    return float(np.max(np.abs(displacement[:4])))


def build_error_groups(analysis_data: dict, dataset_root: Path) -> tuple[list[dict], dict]:
    by_variant = analysis_data["by_variant"]
    retrieval = analysis_data["retrieval_rows_by_score"]
    anchor_aggregates = analysis_data["anchor_aggregates"]
    errors = {}
    ready = {}
    for variant_id in VARIANT_IDS_FOR_GROUPS:
        rows = sorted(by_variant[variant_id], key=lambda row: int(row["pair_index"]))
        errors[variant_id] = np.asarray([as_float(row["pair_public_error"]) for row in rows])
        ready[variant_id] = np.asarray([as_bool(row["pair_local_ready"]) for row in rows], dtype=np.float64)

    cosine_rows = retrieval["cosine"]
    proxies = {
        "visibility_gt_inbounds_fraction": np.asarray(
            [int(row["inbounds_count"]) / 128 for row in cosine_rows], dtype=np.float64
        ),
        "ambiguity_cosine_mean_entropy": np.asarray(
            [np.mean(anchor_aggregates[(i, "cosine")]["entropy"]) for i in range(PAIR_COUNT)]
        ),
        "confidence_cosine_mean_margin": np.asarray(
            [np.mean(anchor_aggregates[(i, "cosine")]["margin"]) for i in range(PAIR_COUNT)]
        ),
        "collision_cosine_unique_target_ratio": np.asarray(
            [as_float(row["unique_target_ratio"]) for row in cosine_rows]
        ),
        "collision_cosine_max_target_reuse": np.asarray(
            [as_float(row["max_target_reuse"]) for row in cosine_rows]
        ),
        "large_transform_corner_linf_px": np.asarray(
            [transform_proxy(dataset_root, pair) for pair in range(PAIR_COUNT)]
        ),
    }
    cosine_top1 = np.asarray(
        [int(row["top1_inlier_count"]) / max(int(row["inbounds_count"]), 1) * 100 for row in cosine_rows]
    )
    cosine_r32 = np.asarray(
        [int(row["recall_num_k32_r1"]) / max(int(row["inbounds_count"]), 1) * 100 for row in cosine_rows]
    )
    cosine_r64_r2 = np.asarray(
        [int(row["recall_num_k64_r2"]) / max(int(row["inbounds_count"]), 1) * 100 for row in cosine_rows]
    )

    group_rows = []
    proxy_summary = {}
    for proxy_index, (proxy_name, values) in enumerate(proxies.items()):
        order = np.argsort(values, kind="stable")
        groups = np.array_split(order, 4)
        proxy_summary[proxy_name] = {
            "definition": "rank quartiles over 424 image pairs; proxy only, not a semantic label",
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }
        for quartile_index, pair_indices in enumerate(groups, start=1):
            row = {
                "proxy": proxy_name,
                "quartile": quartile_index,
                "pair_count": int(len(pair_indices)),
                "proxy_min": float(values[pair_indices].min()),
                "proxy_max": float(values[pair_indices].max()),
                "proxy_mean": float(values[pair_indices].mean()),
                "cosine_top1_inlier_percent": float(cosine_top1[pair_indices].mean()),
                "cosine_R32_r1_percent": float(cosine_r32[pair_indices].mean()),
                "cosine_R64_r2_percent": float(cosine_r64_r2[pair_indices].mean()),
            }
            for variant_offset, variant_id in enumerate(VARIANT_IDS_FOR_GROUPS):
                selected_error = errors[variant_id][pair_indices]
                row[f"{variant_id}:aepe"] = float(selected_error.mean())
                row[f"{variant_id}:aepe_ci"] = independent_group_bootstrap(
                    selected_error, seed=20260804 + proxy_index * 100 + quartile_index * 10 + variant_offset
                )
                row[f"{variant_id}:pair_local_ready_percent"] = float(
                    ready[variant_id][pair_indices].mean() * 100
                )
            group_rows.append(row)
    return group_rows, proxy_summary


def audit_benchmark(root: Path) -> dict:
    aggregate = load_json(root / "aggregate.json")
    gate = load_json(root / "g5_gate.json")
    statuses = load_json(root / "worker_status.json")
    issues = []
    if len(statuses) != 20:
        issues.append(f"worker_count:{len(statuses)}")
    if any(row["return_code"] != 0 or not row["validation"]["passed"] for row in statuses):
        issues.append("worker_status_validation")

    summaries = {}
    worker_checks = {}
    for status in statuses:
        worker_id = status["worker_id"]
        worker_dir = root / "workers" / worker_id
        summary = load_json(worker_dir / "summary.json")
        total_rows = read_jsonl(worker_dir / "timing_total.jsonl")
        summaries[worker_id] = summary
        wall = np.asarray([float(row["wall_ms"]) for row in total_rows])
        iterations = [int(row["iteration"]) for row in total_rows]
        raw_valid = (
            len(total_rows) == 200
            and iterations == list(range(200))
            and np.isfinite(wall).all()
            and all(close(float(row["wall_ms"]), int(row["wall_ns"]) / 1e6, 1e-12) for row in total_rows)
        )
        summary_diff = max(
            abs(float(wall.mean()) - summary["total"]["wall_mean_ms"]),
            abs(float(np.percentile(wall, 50)) - summary["total"]["wall_p50_ms"]),
            abs(float(np.percentile(wall, 95)) - summary["total"]["wall_p95_ms"]),
        )
        segments = read_jsonl(worker_dir / "timing_segments.jsonl")
        segment_names = sorted(summary["segments"])
        segments_valid = (
            len(segments) == 200 * len(segment_names)
            and sorted({row["segment"] for row in segments}) == segment_names
        )
        k_path = worker_dir / "k_scaling_raw.jsonl"
        k_valid = True
        if summary["method"] == "slim":
            k_rows = read_jsonl(k_path)
            k_valid = (
                len(k_rows) == 600
                and sorted({int(row["k"]) for row in k_rows}) == [64, 128, 256]
                and all(np.isfinite(float(row["wall_ms"])) for row in k_rows)
            )
        elif k_path.exists():
            k_valid = False
        expected_shape = [1, 2, 512, 512] if summary["contract"] == "dense-contract" else [1, 3, 3]
        contract_valid = (
            summary["formal_run"]
            and summary["warmup"] == 20
            and summary["repeats"] == 200
            and summary["batch_size"] == 1
            and summary["precision"] == "FP32"
            and summary["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
            and summary["environment"]["inference_mode_enabled"]
            and not summary["environment"]["truth_flow_loaded"]
            and summary["output_dtype"] == "torch.float32"
            and summary["output_shape"] == expected_shape
            and summary["output_finite"]
        )
        passed = raw_valid and summary_diff <= 1e-9 and segments_valid and k_valid and contract_valid
        if not passed:
            issues.append(f"worker_contract:{worker_id}")
        worker_checks[worker_id] = {
            "passed": passed,
            "raw_total_rows": len(total_rows),
            "raw_total_valid": raw_valid,
            "summary_max_abs_diff": summary_diff,
            "segment_rows": len(segments),
            "segment_valid": segments_valid,
            "k_scaling_valid": k_valid,
            "execution_contract_valid": contract_valid,
        }

    recomputed_groups = {}
    for method in ("baseline", "slim"):
        for contract in ("affine-core", "dense-contract"):
            selected = [
                summary for summary in summaries.values()
                if summary["method"] == method and summary["contract"] == contract
            ]
            p50 = np.asarray([summary["total"]["wall_p50_ms"] for summary in selected])
            p95 = np.asarray([summary["total"]["wall_p95_ms"] for summary in selected])
            recomputed_groups[f"{method}:{contract}"] = {
                "replicates": len(selected),
                "process_p50_ms": p50.tolist(),
                "process_p95_ms": p95.tolist(),
                "formal_p50_ms": float(np.median(p50)),
                "formal_p95_ms": float(np.median(p95)),
                "p50_cv_percent": float(np.std(p50, ddof=1) / np.mean(p50) * 100),
                "cold_peak_allocated_mib": float(max(summary["memory"]["mib"]["cold_peak_allocated"] for summary in selected)),
                "warm_peak_allocated_mib": float(max(summary["memory"]["mib"]["warm_peak_allocated"] for summary in selected)),
            }

    slim_dense = recomputed_groups["slim:dense-contract"]
    p50_values = np.asarray(slim_dense["process_p50_ms"])
    leave_one_out = []
    for index in range(len(p50_values)):
        kept = np.delete(p50_values, index)
        leave_one_out.append({
            "excluded_replicate": index,
            "excluded_p50_ms": float(p50_values[index]),
            "remaining_cv_percent": float(np.std(kept, ddof=1) / np.mean(kept) * 100),
        })
    best_exclusion = min(leave_one_out, key=lambda row: row["remaining_cv_percent"])
    segment_profile = {}
    for replicate in range(5):
        summary = summaries[f"dense-contract__r{replicate}__slim"]
        segment_profile[str(replicate)] = {
            name: values["wall_p50_ms"] for name, values in summary["segments"].items()
        }

    k_scaling = aggregate["k_scaling"]
    audit = {
        "passed": not issues,
        "issues": issues,
        "artifact_manifest": verify_manifest(root),
        "worker_checks": worker_checks,
        "recomputed_groups": recomputed_groups,
        "g5_recorded": gate,
        "g5_recomputed_passed": all(gate["checks"].values()),
        "g5_failed_checks": sorted(name for name, passed in gate["checks"].items() if not passed),
        "cv_diagnostic": {
            "leave_one_out": leave_one_out,
            "best_single_exclusion": best_exclusion,
            "segment_p50_ms_by_replicate": segment_profile,
            "interpretation": "diagnostic only; the preregistered five-process CV remains authoritative",
        },
        "k_scaling_diagnostic": {
            "formal_wall_p50_ms": {k: k_scaling[k]["formal_wall_p50_ms"] for k in ("64", "128", "256")},
            "incremental_peak_mib": {k: k_scaling[k]["incremental_peak_allocated_mib_max"] for k in ("64", "128", "256")},
            "timing_fit": k_scaling["timing_linear_fit"],
            "memory_fit": k_scaling["memory_linear_fit"],
            "no_hidden_N2": gate["checks"]["no_hidden_N2"],
            "interpretation": "matcher latency is launch/loop-overhead dominated below 1 ms; this does not override the failed preregistered positive-slope and R2 checks",
        },
    }
    return audit


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    accuracy_audit, bootstrap_ci, analysis_data = audit_accuracy(args.accuracy_dir)
    group_rows, proxy_summary = build_error_groups(analysis_data, args.dataset_root)
    benchmark_audit = audit_benchmark(args.benchmark_dir)

    write_json(args.output_dir / "accuracy_audit.json", accuracy_audit)
    write_json(args.output_dir / "bootstrap_ci.json", bootstrap_ci)
    write_json(args.output_dir / "error_group_summary.json", {
        "label_policy": "All groups are measurable proxies; no occlusion, repeated-texture, or semantic labels are claimed.",
        "proxies": proxy_summary,
        "groups": group_rows,
    })
    write_json(args.output_dir / "benchmark_audit.json", benchmark_audit)
    with (args.output_dir / "error_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        flat_rows = []
        for row in group_rows:
            flat = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat[f"{key}:{subkey}"] = subvalue
                else:
                    flat[key] = value
            flat_rows.append(flat)
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    manifest = {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    write_json(args.output_dir / "artifact_manifest.json", manifest)
    print(json.dumps({
        "accuracy_audit_passed": accuracy_audit["passed"] and accuracy_audit["artifact_manifest"]["passed"],
        "benchmark_audit_passed": benchmark_audit["passed"] and benchmark_audit["artifact_manifest"]["passed"],
        "g5_passed": benchmark_audit["g5_recomputed_passed"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
