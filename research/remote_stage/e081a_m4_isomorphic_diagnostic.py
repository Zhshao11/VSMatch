#!/usr/bin/env python3
"""Frozen E081A-M4 WHU diagnostic with prediction-before-GT separation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
SOURCE = TMP / "e081a_whu_our_native_train.py"
RUN = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-M4-ISOMORPHIC-DIAGNOSTIC/run-01"
E081 = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/run-m4-native-r2"
CHECKPOINT = E081 / "checkpoints/M4_step120000.pt"
EXPECTED_SHA = "4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a"
WHU_VIEW = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/datasets/os_dataset"
SELECTED_ITERS = (12, 24, 32)
TOPK = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_components():
    sys.path.insert(0, str(TMP))
    import e081a_whu_our_native_train as e081

    cfg = e081.make_cfg("E081A-M4-frozen-diagnostic", 1, 1, 512)
    return e081, cfg


def load_model(device: torch.device):
    e081, cfg = load_components()
    model = e081.OurNativeMethod("M4", device, cfg).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return e081, model


def load_images_without_gt(dataset, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    from core.utils import frame_utils

    arrays = []
    for path in dataset.image_list[index]:
        image = np.asarray(frame_utils.read_gen(path)).astype(np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = image[..., :3]
        arrays.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float())
    return arrays[0], arrays[1]


def frozen_forward(model, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32):
    """Exact GDROS recurrence with read-only trace capture."""
    from core.corr import CorrBlock

    backbone = model.backbone
    normalized1 = 2 * (image1 / 255.0) - 1.0
    normalized2 = 2 * (image2 / 255.0) - 1.0
    context = backbone.context_encoder(normalized1)
    fmap1, fmap2 = backbone.featureNet(normalized1, normalized2, {}, context)
    corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=backbone.corr_radius)
    global_corr = corr_fn.corr_pyramid[0].reshape(fmap1.shape[-2] * fmap1.shape[-1], -1)
    global_top32 = global_corr.topk(TOPK, dim=1).indices.to(torch.int32).cpu()

    cnet = backbone.context_encoder(normalized1)
    net, inp = torch.split(cnet, [backbone.hidden_dim, backbone.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    coords0, coords1 = backbone.initialize_flow(normalized1)
    traces, outputs = [], {}
    for iteration in range(1, iters + 1):
        before = coords1.detach()
        corr = corr_fn(before)
        net, up_mask, delta = backbone.update_block(net, inp, corr, before - coords0)
        coords1 = before + delta
        traces.append({
            "iteration": iteration,
            "coords_before": before[0].to(torch.float16).cpu(),
            "coords_after": coords1[0].to(torch.float16).cpu(),
        })
        if iteration in SELECTED_ITERS:
            raw = backbone.upsample_flow(coords1 - coords0, up_mask)
            refined, aux = model.refiner(raw)
            post = e081_lsr(refined)
            outputs[iteration] = {
                "backbone_pre_lsr": raw[0].to(torch.float16).cpu(),
                "m4_pre_lsr": refined[0].to(torch.float16).cpu(),
                "m4_post_lsr": post[0].to(torch.float16).cpu(),
                "aux": aux,
            }
    return {
        "feature_hw": list(fmap1.shape[-2:]),
        "global_top32": global_top32,
        "trajectory": traces,
        "outputs": outputs,
    }


def e081_lsr(flow: torch.Tensor) -> torch.Tensor:
    e081, _ = load_components()
    return e081.lsr_flow(flow).to(flow.device)


def inference() -> None:
    started = time.time()
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("E081A-M4 checkpoint checksum mismatch")
    e081, _ = load_components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="test", root=str(WHU_VIEW), dstype="testing")
    if len(dataset) != 700:
        raise RuntimeError(f"WHU protocol expected 700 pairs, got {len(dataset)}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _, model = load_model(device)
    pred_dir = RUN / "predictions_before_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    manifest = RUN / "PREDICTIONS.jsonl"
    manifest.unlink(missing_ok=True)
    with torch.inference_mode():
        for index in range(len(dataset)):
            image1, image2 = load_images_without_gt(dataset, index)
            pair_id = Path(dataset.image_list[index][0]).stem
            trace = frozen_forward(model, image1[None].to(device), image2[None].to(device))
            path = pred_dir / f"{pair_id}.pt"
            torch.save({"pair_index": index, "pair_id": pair_id, **trace}, path)
            record = {
                "pair_index": index,
                "pair_id": pair_id,
                "prediction_file": str(path),
                "prediction_sha256": sha256(path),
                "uses_ground_truth": False,
                "checkpoint_sha256": EXPECTED_SHA,
            }
            with manifest.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if index % 10 == 0:
                print(json.dumps({"stage": "inference", "done": index + 1, "total": len(dataset)}), flush=True)
            torch.cuda.empty_cache()
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 700 or len(list(pred_dir.glob("*.pt"))) != 700:
        raise RuntimeError("prediction materialization incomplete")
    complete = {
        "status": "COMPLETE",
        "last_completed_stage": "ALL_700_PREDICTIONS_AND_NON_GT_TRACES_MATERIALIZED",
        "pairs": 700,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_SHA,
        "prediction_materialized_before_gt_join": True,
        "candidate_gpus": None,
        "logical_device": "cuda:0",
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN / "INFERENCE_RESULT_CARD.json", complete)
    write_json(RUN / "RESULT_CARD.json", complete)


def parity() -> None:
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("E081A-M4 checkpoint checksum mismatch")
    e081, _ = load_components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="test", root=str(WHU_VIEW), dstype="testing")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _, model = load_model(device)
    image1, image2 = load_images_without_gt(dataset, 0)
    image1, image2 = image1[None].to(device), image2[None].to(device)
    with torch.inference_mode():
        production = model(image1, image2, iters=32)
        captured = frozen_forward(model, image1, image2)
    comparisons = []
    for iteration in SELECTED_ITERS:
        expected = production[iteration - 1].float().cpu()
        actual = captured["outputs"][iteration]["m4_pre_lsr"][None].float()
        difference = (expected - actual).abs()
        comparisons.append({
            "iteration": iteration,
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "shape": list(expected.shape),
        })
    status = "PASS" if all(row["max_abs"] <= 0.04 for row in comparisons) else "FAIL"
    card = {
        "status": status,
        "last_completed_stage": "FROZEN_RECURRENCE_PARITY",
        "checkpoint_sha256": EXPECTED_SHA,
        "pair_index": 0,
        "comparisons": comparisons,
        "tolerance": "0.04 px after float16 trace serialization; observed quantization max is 0.03125 px",
        "architecture_audit": {
            "F2": False,
            "F3": False,
            "B": False,
            "C": False,
            "M4": "single GDROS/RAFT backbone plus region NativeGateRefiner",
        },
    }
    write_json(RUN / "P0_PARITY_RESULT_CARD.json", card)
    write_json(RUN / "RESULT_CARD.json", card)
    print(json.dumps(card, indent=2, sort_keys=True))
    if status != "PASS":
        raise RuntimeError(f"parity failed: {comparisons}")


def pair_epe(flow: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(flow.float() - gt.float(), dim=0)
    return float(error[valid].mean())


def local_oracle_epe(flow: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, radius: int) -> float:
    residual = gt.float() - flow.float()
    correction = residual.round().clamp(-radius, radius)
    error = torch.linalg.vector_norm(flow.float() + correction - gt.float(), dim=0)
    return float(error[valid].mean())


def metric_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "pairs": int(array.size),
        "aepe": float(array.mean()),
        **{f"cmr{threshold}": float(100 * (array < threshold).mean()) for threshold in (1, 2, 5, 10, 16)},
    }


def bucket(value: float) -> str:
    if value < 1:
        return "<1"
    if value < 2:
        return "[1,2)"
    if value < 5:
        return "[2,5)"
    if value < 10:
        return "[5,10)"
    if value < 16:
        return "[10,16)"
    return ">=16"


def join() -> None:
    started = time.time()
    inference_card = json.loads((RUN / "INFERENCE_RESULT_CARD.json").read_text())
    manifest_path = Path(inference_card["manifest"])
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    pred_files = sorted((RUN / "predictions_before_gt").glob("*.pt"))
    if inference_card["status"] != "COMPLETE" or len(records) != 700 or len(pred_files) != 700:
        raise RuntimeError("GT join blocked until all predictions are materialized")
    if sha256(manifest_path) != inference_card["manifest_sha256"]:
        raise RuntimeError("prediction manifest changed before GT join")

    e081, _ = load_components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="testing", root=str(WHU_VIEW), dstype="testing")
    rows, failure_rows = [], []
    for record in records:
        index = int(record["pair_index"])
        file_opt, _image1, _image2, gt, valid = dataset[index]
        pair_id = Path(file_opt).stem
        if pair_id != record["pair_id"]:
            raise RuntimeError(f"pair order mismatch at {index}")
        data = torch.load(record["prediction_file"], map_location="cpu")
        valid_mask = valid.bool()
        row = {"pair_id": pair_id, "pair_index": index}
        for iteration in SELECTED_ITERS:
            outputs = data["outputs"][iteration]
            for name in ("backbone_pre_lsr", "m4_pre_lsr", "m4_post_lsr"):
                row[f"iter{iteration}_{name}"] = pair_epe(outputs[name], gt, valid_mask)
        final_pre = data["outputs"][32]["m4_pre_lsr"]
        final_post = data["outputs"][32]["m4_post_lsr"]
        for stage_name, flow in (("pre_lsr", final_pre), ("post_lsr", final_post)):
            for radius in (1, 2, 4):
                row[f"{stage_name}_local_oracle_r{radius}"] = local_oracle_epe(flow, gt, valid_mask, radius)

        h, w = data["feature_hw"]
        ys = torch.arange(h) * 8
        xs = torch.arange(w) * 8
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        sampled = gt[:, gy, gx].permute(1, 2, 0) / 8.0
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        target = torch.stack((grid_x, grid_y), dim=-1).float() + sampled
        target_index = target[..., 1].round().long().clamp(0, h - 1) * w + target[..., 0].round().long().clamp(0, w - 1)
        target_index = target_index.flatten()
        top32 = data["global_top32"].long()
        for k in (1, 2, 4, 8, 16, 32):
            row[f"global_corr_top{k}_gt_coverage"] = float(100 * (top32[:, :k] == target_index[:, None]).any(dim=1).float().mean())

        trajectory = []
        target_flat = target.reshape(-1, 2)
        for trace in data["trajectory"]:
            before = trace["coords_before"].permute(1, 2, 0).reshape(-1, 2).float()
            after = trace["coords_after"].permute(1, 2, 0).reshape(-1, 2).float()
            dist_before = torch.linalg.vector_norm(before - target_flat, dim=1) * 8
            dist_after = torch.linalg.vector_norm(after - target_flat, dim=1) * 8
            delta = target_flat - before
            levels = [((delta[:, 0].abs() / (2**level) <= 4) & (delta[:, 1].abs() / (2**level) <= 4)) for level in range(4)]
            trajectory.append({
                "iteration": trace["iteration"],
                "mean_gt_distance_before": float(dist_before.mean()),
                "mean_gt_distance_after": float(dist_after.mean()),
                "update_toward_gt_percent": float(100 * (dist_after < dist_before).float().mean()),
                "gt_in_any_radius4_pyramid_percent": float(100 * torch.stack(levels).any(dim=0).float().mean()),
                **{f"gt_in_level{level}_radius4_percent": float(100 * levels[level].float().mean()) for level in range(4)},
            })
        row["trajectory"] = trajectory
        row["final_bucket"] = bucket(row["iter32_m4_post_lsr"])
        rows.append(row)
        if row["iter32_m4_post_lsr"] >= 5:
            failure_rows.append(row)
        if index % 20 == 0:
            print(json.dumps({"stage": "gt_join", "done": index + 1, "total": len(records)}), flush=True)

    flat_fields = [key for key in rows[0] if key != "trajectory"]
    with (RUN / "PAIR_LEVEL_METRIC_JOIN.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_fields} for row in rows)
    with (RUN / "FAILURE_SAMPLES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_fields} for row in failure_rows)
    with (RUN / "PAIR_LEVEL_DIAGNOSTICS.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metric_keys = [f"iter{iteration}_{name}" for iteration in SELECTED_ITERS for name in ("backbone_pre_lsr", "m4_pre_lsr", "m4_post_lsr")]
    metric_keys += [f"{stage}_local_oracle_r{radius}" for stage in ("pre_lsr", "post_lsr") for radius in (1, 2, 4)]
    metrics = {key: metric_summary([row[key] for row in rows]) for key in metric_keys}
    bucket_counts = {name: sum(row["final_bucket"] == name for row in rows) for name in ("<1", "[1,2)", "[2,5)", "[5,10)", "[10,16)", ">=16")}
    trajectory_summary = []
    for iteration in range(1, 33):
        values = [row["trajectory"][iteration - 1] for row in rows]
        trajectory_summary.append({
            "iteration": iteration,
            **{key: float(np.mean([value[key] for value in values])) for key in values[0] if key != "iteration"},
        })
    evidence = {
        f"top{k}_mean_gt_coverage_percent": float(np.mean([row[f"global_corr_top{k}_gt_coverage"] for row in rows]))
        for k in (1, 2, 4, 8, 16, 32)
    }
    final_failures = len(failure_rows)
    final_traj = trajectory_summary[-1]
    recommendation = []
    if evidence["top32_mean_gt_coverage_percent"] < 50:
        recommendation.append("Prioritize feature/correlation evidence acquisition; GT targets are often absent from global top-32 evidence.")
    if final_traj["gt_in_any_radius4_pyramid_percent"] < 80:
        recommendation.append("Increase basin/search coverage before changing the update rule; GT is frequently outside the recurrent radius-4 pyramid.")
    if final_traj["gt_in_any_radius4_pyramid_percent"] >= 80 and final_traj["update_toward_gt_percent"] < 55:
        recommendation.append("Evidence is locally available but updates under-use it; investigate the recurrent update/context pathway.")
    if metrics["post_lsr_local_oracle_r4"]["aepe"] + 0.2 < metrics["iter32_m4_post_lsr"]["aepe"]:
        recommendation.append("A local fine-localization head has measurable diagnostic headroom at radius 4, subject to a later no-GT design decision.")
    if not recommendation:
        recommendation.append("No single bottleneck dominates; inspect the supplied per-pair overlap before authorizing a new module.")

    result = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "last_completed_stage": "700_PAIR_GT_JOIN_AND_ATTRIBUTION_COMPLETE",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_SHA,
        "protocol_pairs": 700,
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_sha256": inference_card["manifest_sha256"],
        "prediction_materialized_before_gt_join": True,
        "metrics": metrics,
        "branch_decomposition": {
            "requested_F2_F3_B_C_M4": "NOT_OBSERVABLE_IN_E081A_ARCHITECTURE",
            "verified_actual_architecture": "single GDROS/RAFT backbone plus region NativeGateRefiner (M4)",
            "reported_available_decomposition": "backbone_pre_lsr -> M4_refiner_pre_lsr -> post_lsr",
            "no_values_inferred_for_absent_branches": True,
        },
        "fusion_gain_at_iter32": {
            "m4_minus_backbone_aepe": metrics["iter32_m4_pre_lsr"]["aepe"] - metrics["iter32_backbone_pre_lsr"]["aepe"],
            "post_lsr_minus_pre_lsr_aepe": metrics["iter32_m4_post_lsr"]["aepe"] - metrics["iter32_m4_pre_lsr"]["aepe"],
        },
        "global_correlation_evidence": evidence,
        "trajectory_summary": trajectory_summary,
        "final_error_buckets": bucket_counts,
        "failure_count_ge5": final_failures,
        "os_alignment": {
            "same_failure_definition": "final pair-level post-LSR AEPE >= 5 px",
            "os_reference_failure_count": 45,
            "os_reference": "E075D / CURRENT_BEST_OS_METHOD_20260913",
            "comparison_scope": "diagnostic counts only; datasets and architectures differ",
        },
        "artifacts": {
            "pair_join": str(RUN / "PAIR_LEVEL_METRIC_JOIN.csv"),
            "pair_diagnostics": str(RUN / "PAIR_LEVEL_DIAGNOSTICS.jsonl"),
            "failure_samples": str(RUN / "FAILURE_SAMPLES.csv"),
        },
        "next_method_recommendation": recommendation,
        "ground_truth_policy": "GT used only after complete prediction materialization, for metric join and explicitly labeled diagnostics/oracles; never for formal prediction selection.",
        "training_or_new_module_started": False,
        "lucky_r2_started": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN / "RESULT_CARD.json", result)
    (RUN / "RESULT_CARD.md").write_text(
        "# E081A-M4 WHU isomorphic diagnostic\n\n"
        "Status: `COMPLETE_DIAGNOSTIC_ONLY`\n\n"
        f"700-pair iter32 M4 post-LSR: AEPE `{metrics['iter32_m4_post_lsr']['aepe']:.6f}`, "
        f"CMR@1/2/5/10/16 `{metrics['iter32_m4_post_lsr']['cmr1']:.4f}/{metrics['iter32_m4_post_lsr']['cmr2']:.4f}/{metrics['iter32_m4_post_lsr']['cmr5']:.4f}/{metrics['iter32_m4_post_lsr']['cmr10']:.4f}/{metrics['iter32_m4_post_lsr']['cmr16']:.4f}`.\n\n"
        f"Final >=5 px failures: `{final_failures}`. Predictions and traces were fully materialized before the separate GT join.\n\n"
        "E081A-M4 contains no F2/F3 or B/C branches. The available exact decomposition is backbone -> region refiner -> LSR; absent branch values were not inferred.\n\n"
        "No training, new module, checkpoint selection, or LuckyR2 run was performed.\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parity", "inference", "join"), required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "parity":
            parity()
        elif args.mode == "inference":
            inference()
        else:
            join()
    except BaseException as error:
        card = {
            "status": "IMPLEMENTATION_BLOCKED",
            "mode": args.mode,
            "cause": repr(error),
            "traceback": traceback.format_exc(),
        }
        write_json(RUN / f"{args.mode.upper()}_FAILURE_RESULT_CARD.json", card)
        write_json(RUN / "RESULT_CARD.json", card)
        raise


if __name__ == "__main__":
    main()
