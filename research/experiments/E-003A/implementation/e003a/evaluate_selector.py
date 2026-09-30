#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.sparsegeo.e002a.diagnostics import binary_ranking_metrics
from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002b.candidates import CandidateSet
from experiments.sparsegeo.e002b.consensus import streaming_consensus_search
from experiments.sparsegeo.e002b.reranker import reassign_and_refit
from experiments.sparsegeo.e002b.triple_bank import generate_triple_bank
from experiments.sparsegeo.e003a.candidate_cache import read_flow
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest, sha256_file
from experiments.sparsegeo.e003a.provenance import runtime_identity
from experiments.sparsegeo.e003a.selector import (
    LearnedGeoSelector,
    affine_flow_at_pixel_points,
    dense_affine_flow,
    nonlearned_soft_em,
    weighted_affine_solve,
)


def _load_candidate(cache_root, split, pair_id):
    return torch.load(
        cache_root / "candidates" / split / f"{pair_id}.pt",
        map_location="cpu",
        weights_only=True,
    )


def _load_labels(cache_root, split, pair_id):
    return torch.load(
        cache_root / "labels" / split / f"{pair_id}.pt",
        map_location="cpu",
        weights_only=True,
    )


def _pair_metrics(affine, flow_truth, grid_xy, grid_truth):
    dense = dense_affine_flow(affine)[0].cpu()
    error = torch.linalg.vector_norm(dense - flow_truth, dim=-1)
    grid_prediction = affine_flow_at_pixel_points(affine, grid_xy.unsqueeze(0))[0].cpu()
    grid_error_xy = grid_prediction - grid_truth.cpu()
    grid_l2 = torch.linalg.vector_norm(grid_error_xy, dim=-1)
    grid_linf = grid_error_xy.abs().amax(dim=-1).reshape(16, 16)
    corners = torch.stack((grid_linf[0, 0], grid_linf[0, -1], grid_linf[-1, 0], grid_linf[-1, -1]))
    geometry = {
        "grid_mean_l2_px": float(grid_l2.mean().item()),
        "grid_p95_l2_px": float(torch.quantile(grid_l2, 0.95).item()),
        "corner_max_linf_px": float(corners.max().item()),
        "capture_at_32_percent": float(100.0 * (grid_linf <= 32.0).float().mean().item()),
    }
    geometry["pair_local_ready"] = bool(
        geometry["grid_p95_l2_px"] <= 24.0
        and geometry["corner_max_linf_px"] <= 32.0
        and geometry["capture_at_32_percent"] >= 90.0
    )
    return float(error.mean().item()), geometry


def _dense_pair_metrics(dense_flow, flow_truth, grid_xy, grid_truth):
    if dense_flow.shape == (1, 2, 512, 512):
        dense = dense_flow[0].permute(1, 2, 0).cpu()
    elif dense_flow.shape == (1, 512, 512, 2):
        dense = dense_flow[0].cpu()
    else:
        raise ValueError(f"unexpected dense flow shape: {tuple(dense_flow.shape)}")
    error = torch.linalg.vector_norm(dense - flow_truth, dim=-1)
    grid_norm = 2.0 * grid_xy.to(torch.float32) / 511.0 - 1.0
    sampled = torch.nn.functional.grid_sample(
        dense.permute(2, 0, 1).unsqueeze(0).to(grid_xy.device),
        grid_norm.reshape(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(2, -1).transpose(0, 1)
    grid_error_xy = sampled - grid_truth
    grid_l2 = torch.linalg.vector_norm(grid_error_xy, dim=-1)
    grid_linf = grid_error_xy.abs().amax(dim=-1).reshape(16, 16)
    corners = torch.stack((grid_linf[0, 0], grid_linf[0, -1], grid_linf[-1, 0], grid_linf[-1, -1]))
    geometry = {
        "grid_mean_l2_px": float(grid_l2.mean().item()),
        "grid_p95_l2_px": float(torch.quantile(grid_l2, 0.95).item()),
        "corner_max_linf_px": float(corners.max().item()),
        "capture_at_32_percent": float(100.0 * (grid_linf <= 32.0).float().mean().item()),
    }
    geometry["pair_local_ready"] = bool(
        geometry["grid_p95_l2_px"] <= 24.0
        and geometry["corner_max_linf_px"] <= 32.0
        and geometry["capture_at_32_percent"] >= 90.0
    )
    return float(error.mean().item()), geometry


def _frozen_e002b_control(candidate, anchors, bank, device):
    candidates = CandidateSet(
        score_type="cosine",
        indices=candidate["candidate_indices"].unsqueeze(0).to(device),
        scores=candidate["candidate_scores"].unsqueeze(0).to(device),
        xy=candidate["target_xy_token"].unsqueeze(0).to(device),
    )
    source_xy = anchors.lattice_xy.to(device, torch.float32)
    search = streaming_consensus_search(source_xy, candidates, candidates, bank)
    return reassign_and_refit(search.full_top, source_xy, candidates)


def _summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "aepe": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "cmr1_percent": float(100.0 * np.mean(array < 1.0)),
        "cmr2_percent": float(100.0 * np.mean(array < 2.0)),
        "cmr5_percent": float(100.0 * np.mean(array < 5.0)),
        "failure_ge5_percent": float(100.0 * np.mean(array >= 5.0)),
    }


def _bootstrap(values, ready, seed=20260805, replicates=2000):
    values = np.asarray(values, dtype=np.float64)
    ready = np.asarray(ready, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(replicates, values.size))
    return {
        "aepe_95_ci": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist(),
        "pair_local_ready_95_ci": (
            100.0 * np.quantile(ready[indices].mean(axis=1), [0.025, 0.975])
        ).tolist(),
    }


def _risk_coverage_curve(errors, selection_score):
    errors = np.asarray(errors, dtype=np.float64)
    selection_score = np.asarray(selection_score, dtype=np.float64)
    order = np.argsort(selection_score, kind="mergesort")
    rows = []
    for coverage_percent in range(10, 101, 10):
        count = math.ceil(coverage_percent / 100.0 * errors.size)
        rows.append(
            {
                "coverage_percent": coverage_percent,
                "pair_count": count,
                "selective_aepe": float(errors[order[:count]].mean()),
            }
        )
    return rows


def _diagnostic_attributes(record, labels):
    grid_xy = labels["grid_xy_pixel"].reshape(16, 16, 2).to(torch.float32)
    mapped = grid_xy + labels["grid_truth_flow"].reshape(16, 16, 2).to(torch.float32)
    displacement = torch.linalg.vector_norm(labels["grid_truth_flow"], dim=-1).mean().item()
    step = 511.0 / 15.0
    derivative_x = (mapped[:, 1:] - mapped[:, :-1]).mean(dim=(0, 1)) / step
    derivative_y = (mapped[1:] - mapped[:-1]).mean(dim=(0, 1)) / step
    j11, j21 = derivative_x.tolist()
    j12, j22 = derivative_y.tolist()
    rotation = abs(math.degrees(math.atan2(j21 - j12, j11 + j22)))
    scale = math.sqrt(abs(j11 * j22 - j12 * j21))
    scale_log2_abs = abs(math.log2(max(scale, 1e-8)))
    out_of_bounds = (
        (mapped[..., 0] < 0)
        | (mapped[..., 0] > 511)
        | (mapped[..., 1] < 0)
        | (mapped[..., 1] > 511)
    ).float().mean().item()
    recoverable = labels["recoverable"].bool()
    ranks = labels["candidate_distance_gt"].argmin(dim=-1)[recoverable]
    median_rank = float(ranks.float().median().item()) if ranks.numel() else 8.0
    with Image.open(record["files"]["opt"]["path"]) as image:
        gray = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32).copy())
    texture = float(
        0.5
        * (
            (gray[:, 1:] - gray[:, :-1]).abs().mean()
            + (gray[1:] - gray[:-1]).abs().mean()
        ).item()
    )
    return {
        "displacement_px": float(displacement),
        "rotation_deg": float(rotation),
        "scale_log2_abs": float(scale_log2_abs),
        "black_border_fraction": float(out_of_bounds),
        "candidate_median_rank_0based": median_rank,
        "texture_gradient": texture,
        "group_displacement": "lt32" if displacement < 32 else "32to64" if displacement < 64 else "ge64",
        "group_rotation": "lt5" if rotation < 5 else "5to10" if rotation < 10 else "ge10",
        "group_scale": "lt0.05" if scale_log2_abs < 0.05 else "0.05to0.10" if scale_log2_abs < 0.10 else "ge0.10",
        "group_black_border": "lt1pct" if out_of_bounds < 0.01 else "1to10pct" if out_of_bounds < 0.10 else "ge10pct",
        "group_candidate_rank": "top1" if median_rank < 0.5 else "rank2to3" if median_rank < 2.5 else "rank4to8_or_dustbin",
        "group_texture": "low" if texture < 10 else "medium" if texture < 20 else "high",
    }


def _grouped_main_report(rows):
    report = {}
    for field in (
        "group_displacement",
        "group_rotation",
        "group_scale",
        "group_black_border",
        "group_candidate_rank",
        "group_texture",
    ):
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row["learned-main-T2_pair_epe"])
        report[field[len("group_"):]] = {
            name: {"pair_count": len(values), **_summarize(values)}
            for name, values in sorted(groups.items())
        }
    return report


@torch.no_grad()
def evaluate_dev(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["seed"] != 20260805 or checkpoint.get("variant") != "main":
        raise RuntimeError("formal E-003A main result requires seed 20260805")
    if checkpoint["split_manifest_sha256"] != sha256_file(args.split_manifest):
        raise RuntimeError("main checkpoint split manifest mismatch")
    model = LearnedGeoSelector().to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    unary_checkpoint = torch.load(args.unary_checkpoint, map_location="cpu", weights_only=True)
    if unary_checkpoint["seed"] != 20260805 or unary_checkpoint.get("variant") != "unary":
        raise RuntimeError("formal unary ablation requires its independently trained primary checkpoint")
    if unary_checkpoint["split_manifest_sha256"] != sha256_file(args.split_manifest):
        raise RuntimeError("unary checkpoint split manifest mismatch")
    if not args.smoke:
        for name, content in (("main", checkpoint), ("unary", unary_checkpoint)):
            provenance = content.get("runtime_provenance_identity")
            if provenance != runtime_identity(args.runtime_binding):
                raise RuntimeError(f"{name} checkpoint runtime provenance mismatch")
    unary_model = LearnedGeoSelector().to(device).eval()
    unary_model.load_state_dict(unary_checkpoint["model_state"], strict=True)
    records = load_split_manifest(args.split_manifest, "dev")
    if args.max_pairs is not None:
        if not args.smoke:
            raise ValueError("reduced dev evaluation is smoke-only")
        records = records[: args.max_pairs]
    anchors = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))
    bank = generate_triple_bank(anchors)
    rows = []
    variant_errors = {name: [] for name in ("frozen-e002b-c8-main", "learned-main-T2", "learned-unary-only", "nonlearned-soft-em", "top8-gt-oracle")}
    variant_ready = {name: [] for name in variant_errors}
    variant_grid = {name: [] for name in variant_errors}
    for index, record in enumerate(records):
        pair_id = record["pair_id"]
        candidate = _load_candidate(args.cache_root, "dev", pair_id)
        features = candidate["candidate_features"].unsqueeze(0).to(device)
        source = candidate["source_xy_norm"].unsqueeze(0).to(device)
        target = candidate["target_xy_norm"].unsqueeze(0).to(device)
        scores = candidate["candidate_scores"].unsqueeze(0).to(device)

        main = model(features, source, target, use_feedback=True)
        unary = unary_model(features, source, target, use_feedback=False)
        soft_em_affine, soft_em_condition, soft_em_valid = nonlearned_soft_em(
            scores, source, target
        )
        frozen_b2 = _frozen_e002b_control(candidate, anchors, bank, device)
        # Method outputs are complete before raw truth is loaded for metrics/oracles.
        method_trace = {
            "pair_id": pair_id,
            "uses_ground_truth": False,
            "main_affine": main.affine_normalized.detach().cpu().tolist(),
            "unary_affine": unary.affine_normalized.detach().cpu().tolist(),
            "soft_em_affine": soft_em_affine.detach().cpu().tolist(),
            "frozen_b2_abstained": frozen_b2.abstained,
        }
        labels = _load_labels(args.cache_root, "dev", pair_id)
        best = labels["candidate_distance_gt"].argmin(dim=-1).to(device)
        oracle_weights = torch.zeros(1, 128, 8, device=device)
        oracle_weights.scatter_(2, best.view(1, 128, 1), 1.0)
        oracle_affine, oracle_condition, oracle_valid = weighted_affine_solve(
            source, target, oracle_weights
        )
        flow_truth = read_flow(Path(record["files"]["truth_flow"]["path"]))
        grid_xy = labels["grid_xy_pixel"].to(device)
        grid_truth = labels["grid_truth_flow"].to(device)
        variants = {
            "learned-main-T2": (main.affine_normalized, bool(main.solver_valid.item())),
            "learned-unary-only": (unary.affine_normalized, bool(unary.solver_valid.item())),
            "nonlearned-soft-em": (soft_em_affine, bool(soft_em_valid.item())),
            "top8-gt-oracle": (oracle_affine, bool(oracle_valid.item())),
        }
        metrics = {}
        pair_epe, geometry = _dense_pair_metrics(
            frozen_b2.dense_flow, flow_truth, grid_xy, grid_truth
        )
        variant_errors["frozen-e002b-c8-main"].append(pair_epe)
        variant_ready["frozen-e002b-c8-main"].append(geometry["pair_local_ready"])
        variant_grid["frozen-e002b-c8-main"].append(geometry["grid_mean_l2_px"])
        metrics["frozen-e002b-c8-main"] = {
            "pair_epe": pair_epe,
            "solver_valid": not frozen_b2.abstained,
            **geometry,
        }
        for name, (affine, valid) in variants.items():
            pair_epe, geometry = _pair_metrics(affine, flow_truth, grid_xy, grid_truth)
            if not math.isfinite(pair_epe):
                raise RuntimeError(f"non-finite metric output for {pair_id} {name}")
            variant_errors[name].append(pair_epe)
            variant_ready[name].append(geometry["pair_local_ready"])
            variant_grid[name].append(geometry["grid_mean_l2_px"])
            metrics[name] = {"pair_epe": pair_epe, "solver_valid": valid, **geometry}
        raw_predicted = float(main.predicted_grid_error_px.item())
        confidence_finite = math.isfinite(raw_predicted)
        main_abstain = bool(main.abstain.item()) or not confidence_finite
        selection_score = raw_predicted if confidence_finite and bool(main.solver_valid.item()) else 1e30
        diagnostics = _diagnostic_attributes(record, labels)
        rows.append(
            {
                "pair_id": pair_id,
                "predicted_grid_error_px": raw_predicted if confidence_finite else None,
                "confidence_selection_score": selection_score,
                "confidence_finite": confidence_finite,
                "solver_valid": bool(main.solver_valid.item()),
                "abstain": main_abstain,
                "condition": float(main.condition.item()),
                "method_trace_uses_ground_truth": method_trace["uses_ground_truth"],
                **diagnostics,
                **{
                    f"{name}_{key}": value
                    for name, values in metrics.items()
                    for key, value in values.items()
                },
            }
        )
        if index % args.log_every == 0:
            print(f"dev {index}/{len(records)} {pair_id}", flush=True)

    summaries = {}
    for name in variant_errors:
        summaries[name] = {
            **_summarize(variant_errors[name]),
            "pair_local_ready_percent": float(100.0 * np.mean(variant_ready[name])),
            "grid_mean_l2_px_mean": float(np.mean(variant_grid[name])),
        }
    main_errors = np.asarray(variant_errors["learned-main-T2"])
    predicted = np.asarray([row["confidence_selection_score"] for row in rows])
    abstain = np.asarray([row["abstain"] for row in rows], dtype=bool)
    coverage = float(100.0 * np.mean(~abstain))
    count90 = math.ceil(0.9 * len(rows))
    selected = np.argsort(predicted, kind="mergesort")[:count90]
    risk90 = float(main_errors[selected].mean())
    no_reject = float(main_errors.mean())
    reduction = float(100.0 * (no_reject - risk90) / no_reject) if no_reject > 0 else 0.0
    ranking = binary_ranking_metrics(main_errors < 5.0, -predicted)
    main_summary = summaries["learned-main-T2"]
    checks = {
        "aepe_le_3": main_summary["aepe"] <= 3.0,
        "pair_local_ready_ge_90": main_summary["pair_local_ready_percent"] >= 90.0,
        "grid_mean_le_8": main_summary["grid_mean_l2_px_mean"] <= 8.0,
        "non_abstention_ge_90": coverage >= 90.0,
        "selective_risk_reduction_ge_20": reduction >= 20.0,
        "all_finite_or_explicit_abstention": bool(
            np.isfinite(main_errors).all()
            and all(row["confidence_finite"] or row["abstain"] for row in rows)
        ),
    }
    result = {
        "experiment": "E-003A",
        "gate": "G1_DEV_ACCURACY_CONFIDENCE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pair_count": len(rows),
        "smoke": args.smoke,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "unary_checkpoint": str(args.unary_checkpoint.resolve()),
        "unary_checkpoint_sha256": sha256_file(args.unary_checkpoint),
        "variants": summaries,
        "confidence": {
            "non_abstention_percent": coverage,
            "risk_at_fixed_90_percent_coverage": risk90,
            "no_abstention_risk": no_reject,
            "risk_reduction_percent": reduction,
            "ranking": ranking,
            "risk_coverage_curve": _risk_coverage_curve(main_errors, predicted),
        },
        "oracle_gap": {
            "aepe_px": main_summary["aepe"] - summaries["top8-gt-oracle"]["aepe"],
            "pair_local_ready_percentage_points": (
                main_summary["pair_local_ready_percent"]
                - summaries["top8-gt-oracle"]["pair_local_ready_percent"]
            ),
        },
        "diagnostic_group_definitions": {
            "displacement_px": ["lt32", "32to64", "ge64"],
            "rotation_deg_from_mean_grid_jacobian": ["lt5", "5to10", "ge10"],
            "absolute_log2_scale_from_mean_grid_jacobian": ["lt0.05", "0.05to0.10", "ge0.10"],
            "raw_flow_mapped_out_of_bounds_fraction": ["lt1pct", "1to10pct", "ge10pct"],
            "median_recoverable_top8_rank": ["top1", "rank2to3", "rank4to8_or_dustbin"],
            "optical_grayscale_mean_absolute_gradient": ["low_lt10", "medium_10to20", "high_ge20"],
            "used_for_gate_or_selection": False
        },
        "diagnostic_groups": _grouped_main_report(rows),
        "bootstrap": _bootstrap(main_errors, variant_ready["learned-main-T2"]),
        "checks": checks,
        "gt_isolation": {
            "method_outputs_before_truth_load": True,
            "all_method_trace_uses_ground_truth_false": all(
                not row["method_trace_uses_ground_truth"] for row in rows
            ),
            "datum_used": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "g1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = args.output_dir / "per_pair.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": result["status"], "checks": checks, "main": main_summary}, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unary-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--cache-gate", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    from experiments.sparsegeo.e003a.spec_binding import validate_frozen_specs
    from experiments.sparsegeo.e003a.provenance import validate_runtime_bindings

    validate_frozen_specs()
    arguments = parse_args()
    arguments.runtime_binding = validate_runtime_bindings(
        arguments.repo_root,
        arguments.run_manifest,
        arguments.formal_config,
        arguments.split_manifest,
        arguments.cache_gate,
        cache_root=arguments.cache_root,
        cache_pair_ids=[row["pair_id"] for row in load_split_manifest(arguments.split_manifest, "dev")],
    )
    evaluate_dev(arguments)
