#!/usr/bin/env python3
"""P0: frozen, no-GT local-candidate discernibility for E081A-M4 on WHU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
RUN = ROOT / "GDROS/experiments/sparsegeo/runs/E-088A-WHU-NOGT-LOCAL-REFINEMENT/P0-candidate-discernibility"
E081 = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/run-m4-native-r2"
CHECKPOINT = E081 / "checkpoints/M4_step120000.pt"
EXPECTED_SHA = "4122c27a7a0a1d0cf6488b9207277b0024d2fe0a9f07f6fcd8eec8ef09ca745a"
WHU_VIEW = ROOT / "GDROS/experiments/sparsegeo/runs/E-081A-ZZVR-WHU-OUR-METHOD-NATIVE-TRAIN/datasets/os_dataset"
PREDICTIONS = RUN / "predictions_before_gt"
OFFSETS = [(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)]
TEMPERATURE = 0.07


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


def components():
    sys.path.insert(0, str(TMP))
    import e081a_whu_our_native_train as e081

    cfg = e081.make_cfg("E088A-P0-frozen-E081A-M4", 1, 1, 512)
    return e081, cfg


def load_model(device: torch.device):
    e081, cfg = components()
    model = e081.OurNativeMethod("M4", device, cfg).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return e081, model


def read_images_without_gt(dataset, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    from core.utils import frame_utils

    result = []
    for path in dataset.image_list[index]:
        image = np.asarray(frame_utils.read_gen(path)).astype(np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = image[..., :3]
        result.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float())
    return result[0], result[1]


def sample_field(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Sample BCHW field at pixel/feature coordinates shaped N,K,2."""
    height, width = field.shape[-2:]
    grid = coordinates.clone()
    grid[..., 0] = 2 * grid[..., 0] / max(width - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(height - 1, 1) - 1
    sampled = F.grid_sample(field, grid[None], mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0].permute(1, 2, 0)


def frozen_anchor_and_features(model, image1: torch.Tensor, image2: torch.Tensor):
    from core.corr import CorrBlock

    backbone = model.backbone
    normalized1 = 2 * (image1 / 255.0) - 1.0
    normalized2 = 2 * (image2 / 255.0) - 1.0
    context = backbone.context_encoder(normalized1)
    fmap1, fmap2 = backbone.featureNet(normalized1, normalized2, {}, context)
    corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=backbone.corr_radius)
    cnet = backbone.context_encoder(normalized1)
    net, inp = torch.split(cnet, [backbone.hidden_dim, backbone.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    coords0, coords1 = backbone.initialize_flow(normalized1)
    final = None
    for _ in range(32):
        coords1 = coords1.detach()
        corr = corr_fn(coords1)
        net, mask, delta = backbone.update_block(net, inp, corr, coords1 - coords0)
        coords1 = coords1 + delta
        final = backbone.upsample_flow(coords1 - coords0, mask)
    refined, _aux = model.refiner(final)
    anchor = e081_lsr(refined)
    return anchor, fmap1.float(), fmap2.float()


def e081_lsr(flow: torch.Tensor) -> torch.Tensor:
    e081, _ = components()
    return e081.lsr_flow(flow).to(flow.device)


def neighbor_mean(flow: torch.Tensor) -> torch.Tensor:
    # flow is H,W,2 on the stride-8 source lattice.
    value = flow.permute(2, 0, 1)[None]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=flow.device,
    )[None, None]
    numerator = F.conv2d(value, kernel.expand(2, 1, 3, 3), padding=1, groups=2)
    denominator = F.conv2d(torch.ones_like(value[:, :1]), kernel, padding=1).clamp_min(1)
    return (numerator / denominator).reshape(2, -1).T


def candidate_bundle(model, image1: torch.Tensor, image2: torch.Tensor) -> dict:
    forward, source_features, target_features = frozen_anchor_and_features(model, image1, image2)
    reverse, _, _ = frozen_anchor_and_features(model, image2, image1)
    _, _, height, width = source_features.shape
    ys = torch.arange(height, device=image1.device)
    xs = torch.arange(width, device=image1.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    source_pixels = torch.stack((grid_x, grid_y), dim=-1).float() * 8
    source_flat = source_pixels.reshape(-1, 2)
    anchor_flow = forward[0, :, grid_y * 8, grid_x * 8].permute(1, 2, 0).reshape(-1, 2)
    target_pixels = source_flat + anchor_flow
    offsets = torch.tensor(OFFSETS, dtype=torch.float32, device=image1.device)
    candidate_target_pixels = target_pixels[:, None, :] + offsets[None]
    target_feature_coordinates = candidate_target_pixels / 8.0

    source_norm = F.normalize(source_features[0].reshape(source_features.shape[1], -1).T, dim=1)
    target_norm = F.normalize(target_features, dim=1)
    candidate_target_features = sample_field(target_norm, target_feature_coordinates)
    correlation = torch.einsum("nc,nkc->nk", source_norm, candidate_target_features)

    reverse_at_candidates = sample_field(reverse, candidate_target_pixels)
    candidate_displacements = anchor_flow[:, None, :] + offsets[None]
    cycle_error = torch.linalg.vector_norm(candidate_displacements + reverse_at_candidates, dim=2)
    local_mean = neighbor_mean(anchor_flow.reshape(height, width, 2))
    neighbor_error = torch.linalg.vector_norm(candidate_displacements - local_mean[:, None, :], dim=2)

    probabilities = torch.softmax(correlation / TEMPERATURE, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=1) / math.log(len(OFFSETS))
    top = correlation.topk(8, dim=1)
    return {
        "feature_hw": [height, width],
        "anchor_flow": anchor_flow.to(torch.float16).cpu(),
        "correlation": correlation.to(torch.float16).cpu(),
        "cycle_error": cycle_error.to(torch.float16).cpu(),
        "neighbor_error": neighbor_error.to(torch.float16).cpu(),
        "score_top8_indices": top.indices.to(torch.uint8).cpu(),
        "score_top8_values": top.values.to(torch.float16).cpu(),
        "score_margin": (top.values[:, 0] - top.values[:, 1]).to(torch.float16).cpu(),
        "score_entropy": entropy.to(torch.float16).cpu(),
        "offsets": OFFSETS,
        "uses_ground_truth": False,
    }


def smoke() -> None:
    started = time.time()
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("protected E081A-M4 checkpoint hash mismatch")
    e081, _ = components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="test", root=str(WHU_VIEW), dstype="testing")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _, model = load_model(device)
    image1, image2 = read_images_without_gt(dataset, 0)
    with torch.inference_mode():
        bundle = candidate_bundle(model, image1[None].to(device), image2[None].to(device))
    expected_tokens = bundle["feature_hw"][0] * bundle["feature_hw"][1]
    checks = {
        "anchor_shape": list(bundle["anchor_flow"].shape) == [expected_tokens, 2],
        "correlation_shape": list(bundle["correlation"].shape) == [expected_tokens, 81],
        "cycle_shape": list(bundle["cycle_error"].shape) == [expected_tokens, 81],
        "neighbor_shape": list(bundle["neighbor_error"].shape) == [expected_tokens, 81],
        "finite": all(torch.isfinite(bundle[key]).all().item() for key in ("anchor_flow", "correlation", "cycle_error", "neighbor_error", "score_margin", "score_entropy")),
        "uses_ground_truth_false": bundle["uses_ground_truth"] is False,
        "offsets_fixed_81": bundle["offsets"] == OFFSETS,
    }
    card = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "last_completed_stage": "P0_NO_GT_CANDIDATE_SMOKE",
        "checkpoint_sha256": EXPECTED_SHA,
        "pair_index": 0,
        "feature_hw": bundle["feature_hw"],
        "checks": checks,
        "candidate_gpus": None,
        "logical_device": "cuda:0",
        "truth_flow_opened": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN / "P0_SMOKE_RESULT_CARD.json", card)
    (RUN / "P0_SMOKE_RESULT_CARD.md").write_text(
        "# E088A P0 no-GT smoke\n\n"
        f"Status: `{card['status']}`\n\nChecks: `{checks}`\n\nNo truth flow was opened.\n"
    )
    write_json(RUN / "RESULT_CARD.json", card)
    if card["status"] != "PASS":
        raise RuntimeError(f"P0 smoke checks failed: {checks}")


def inference() -> None:
    started = time.time()
    state = {"status": "RUNNING", "stage": "START", "started_at_unix": started}
    write_json(RUN / "P0_INFERENCE_RESULT_CARD.json", state)
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("protected E081A-M4 checkpoint hash mismatch")
    e081, _ = components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="test", root=str(WHU_VIEW), dstype="testing")
    if len(dataset) != 700:
        raise RuntimeError(f"expected 700 WHU test pairs, got {len(dataset)}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _, model = load_model(device)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(len(dataset)):
            pair_id = Path(dataset.image_list[index][0]).stem
            path = PREDICTIONS / f"{pair_id}.pt"
            if path.exists():
                continue
            image1, image2 = read_images_without_gt(dataset, index)
            bundle = candidate_bundle(model, image1[None].to(device), image2[None].to(device))
            torch.save({"pair_index": index, "pair_id": pair_id, **bundle}, path)
            if index % 10 == 0:
                print(json.dumps({"stage": "P0_NO_GT_INFERENCE", "done": index + 1, "total": 700}), flush=True)
            torch.cuda.empty_cache()
    files = sorted(PREDICTIONS.glob("*.pt"))
    if len(files) != 700:
        raise RuntimeError(f"incomplete materialization: {len(files)}/700")
    manifest = RUN / "PREDICTIONS.jsonl"
    with manifest.open("w") as handle:
        for index, path in enumerate(files):
            data = torch.load(path, map_location="cpu")
            handle.write(json.dumps({
                "pair_index": data["pair_index"],
                "pair_id": data["pair_id"],
                "prediction_file": str(path),
                "prediction_sha256": sha256(path),
                "uses_ground_truth": False,
            }, sort_keys=True) + "\n")
    state = {
        "status": "COMPLETE",
        "last_completed_stage": "700_LOCAL_CANDIDATE_BUNDLES_MATERIALIZED_BEFORE_GT",
        "pairs": 700,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_SHA,
        "candidate_definition": "integer target offsets in [-4,4]^2 around frozen iter32 post-LSR anchor; r2 is the centered [-2,2]^2 subset",
        "formal_score": "cosine similarity of frozen E081A high-resolution-side stride-8 source/target features sampled bilinearly at candidate endpoints",
        "temperature_for_entropy": TEMPERATURE,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "prediction_materialized_before_gt_join": True,
        "candidate_gpus": None,
        "logical_device": "cuda:0",
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN / "P0_INFERENCE_RESULT_CARD.json", state)
    (RUN / "P0_INFERENCE_RESULT_CARD.md").write_text(
        "# E088A P0 inference\n\n"
        "Status: `COMPLETE`\n\n700 candidate bundles were materialized without opening truth flow.\n"
    )
    write_json(RUN / "RESULT_CARD.json", state)


def summary(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def pair_metrics(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    return {
        "pairs": int(array.size),
        "aepe": float(array.mean()),
        **{f"cmr{threshold}": float(100 * (array < threshold).mean()) for threshold in (1, 2, 5, 10, 16)},
    }


def join() -> None:
    started = time.time()
    inference_card = json.loads((RUN / "P0_INFERENCE_RESULT_CARD.json").read_text())
    manifest = Path(inference_card["manifest"])
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if inference_card["status"] != "COMPLETE" or len(records) != 700 or sha256(manifest) != inference_card["manifest_sha256"]:
        raise RuntimeError("GT join requires an intact complete 700-pair prediction manifest")
    e081, _ = components()
    _, datasets, *_ = e081.ensure_imports()
    dataset = datasets.opt_sar_test(split="testing", root=str(WHU_VIEW), dstype="testing")
    offsets = torch.tensor(OFFSETS, dtype=torch.float32)
    r2_indices = torch.tensor([i for i, (dx, dy) in enumerate(OFFSETS) if abs(dx) <= 2 and abs(dy) <= 2])
    pair_rows, token_totals = [], {
        "valid": 0,
        "score_top1_oracle_near": 0,
        "score_top5_oracle_near": 0,
        "score_top8_oracle_near": 0,
    }
    oracle_cycle, other_cycle, oracle_neighbor, other_neighbor = [], [], [], []
    for record in records:
        index = int(record["pair_index"])
        file_opt, _image1, _image2, gt, valid = dataset[index]
        pair_id = Path(file_opt).stem
        if pair_id != record["pair_id"]:
            raise RuntimeError(f"pair mismatch at {index}")
        data = torch.load(record["prediction_file"], map_location="cpu")
        height, width = data["feature_hw"]
        ys, xs = torch.arange(height) * 8, torch.arange(width) * 8
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        gt_flow = gt[:, grid_y, grid_x].permute(1, 2, 0).reshape(-1, 2).float()
        valid_tokens = valid[grid_y, grid_x].reshape(-1).bool()
        anchor = data["anchor_flow"].float()
        scores = data["correlation"].float()
        cycle = data["cycle_error"].float()
        neighbor = data["neighbor_error"].float()
        candidate_error = torch.linalg.vector_norm(anchor[:, None, :] + offsets[None] - gt_flow[:, None, :], dim=2)
        candidate_error = candidate_error[valid_tokens]
        scores = scores[valid_tokens]
        cycle = cycle[valid_tokens]
        neighbor = neighbor[valid_tokens]
        anchor_error = torch.linalg.vector_norm(anchor[valid_tokens] - gt_flow[valid_tokens], dim=1)
        oracle4_error, oracle4_index = candidate_error.min(dim=1)
        oracle2_subset_error, oracle2_relative = candidate_error[:, r2_indices].min(dim=1)
        oracle2_index = r2_indices[oracle2_relative]
        ranking = scores.argsort(dim=1, descending=True)
        selected_index = ranking[:, 0]
        selected_error = candidate_error.gather(1, selected_index[:, None]).squeeze(1)
        score_top5_error = candidate_error.gather(1, ranking[:, :5]).min(dim=1).values
        score_top8_error = candidate_error.gather(1, ranking[:, :8]).min(dim=1).values
        # Multiple neighboring candidates can be indistinguishable under continuous GT.
        oracle_near = candidate_error <= oracle4_error[:, None] + 0.25
        top1_near = oracle_near.gather(1, ranking[:, :1]).any(dim=1)
        top5_near = oracle_near.gather(1, ranking[:, :5]).any(dim=1)
        top8_near = oracle_near.gather(1, ranking[:, :8]).any(dim=1)
        token_totals["valid"] += int(valid_tokens.sum())
        token_totals["score_top1_oracle_near"] += int(top1_near.sum())
        token_totals["score_top5_oracle_near"] += int(top5_near.sum())
        token_totals["score_top8_oracle_near"] += int(top8_near.sum())
        token_index = torch.arange(scores.shape[0])
        oracle_cycle.extend(cycle[token_index, oracle4_index].tolist())
        oracle_neighbor.extend(neighbor[token_index, oracle4_index].tolist())
        other_cycle.extend(cycle.mean(dim=1).tolist())
        other_neighbor.extend(neighbor.mean(dim=1).tolist())
        row = {
            "pair_id": pair_id,
            "pair_index": index,
            "valid_tokens": int(valid_tokens.sum()),
            "anchor_aepe": float(anchor_error.mean()),
            "oracle_r2_aepe": float(oracle2_subset_error.mean()),
            "oracle_r4_aepe": float(oracle4_error.mean()),
            "score_top1_aepe": float(selected_error.mean()),
            "score_top5_oracle_aepe": float(score_top5_error.mean()),
            "score_top8_oracle_aepe": float(score_top8_error.mean()),
            "score_top1_oracle_near_percent": float(100 * top1_near.float().mean()),
            "score_top5_oracle_near_percent": float(100 * top5_near.float().mean()),
            "score_top8_oracle_near_percent": float(100 * top8_near.float().mean()),
            "score_margin_mean": float(data["score_margin"][valid_tokens].float().mean()),
            "score_entropy_mean": float(data["score_entropy"][valid_tokens].float().mean()),
            "selected_cycle_error_mean": float(cycle[token_index, selected_index].mean()),
            "selected_neighbor_error_mean": float(neighbor[token_index, selected_index].mean()),
            "oracle_cycle_error_mean": float(cycle[token_index, oracle4_index].mean()),
            "oracle_neighbor_error_mean": float(neighbor[token_index, oracle4_index].mean()),
            "prediction_materialized_before_gt_join": True,
        }
        pair_rows.append(row)
        if index % 25 == 0:
            print(json.dumps({"stage": "P0_GT_JOIN", "done": index + 1, "total": 700}), flush=True)
    with (RUN / "PAIR_LEVEL_METRIC_JOIN.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    methods = {key: pair_metrics([row[key] for row in pair_rows]) for key in (
        "anchor_aepe", "oracle_r2_aepe", "oracle_r4_aepe", "score_top1_aepe", "score_top5_oracle_aepe", "score_top8_oracle_aepe"
    )}
    valid_count = token_totals["valid"]
    retrieval = {
        "score_top1_oracle_near_percent": 100 * token_totals["score_top1_oracle_near"] / valid_count,
        "score_top5_oracle_near_percent": 100 * token_totals["score_top5_oracle_near"] / valid_count,
        "score_top8_oracle_near_percent": 100 * token_totals["score_top8_oracle_near"] / valid_count,
        "random_top1_reference_percent": 100 / 81,
        "random_top5_reference_percent": 500 / 81,
        "random_top8_reference_percent": 800 / 81,
        "oracle_near_definition": "candidate endpoint error <= best r4 candidate error + 0.25 px",
    }
    diagnostics = {
        "score_margin": summary([row["score_margin_mean"] for row in pair_rows]),
        "normalized_score_entropy": summary([row["score_entropy_mean"] for row in pair_rows]),
        "oracle_cycle_error": summary(oracle_cycle),
        "mean_all_candidate_cycle_error": summary(other_cycle),
        "oracle_neighbor_error": summary(oracle_neighbor),
        "mean_all_candidate_neighbor_error": summary(other_neighbor),
    }
    anchor_aepe = methods["anchor_aepe"]["aepe"]
    oracle_gain = anchor_aepe - methods["oracle_r4_aepe"]["aepe"]
    scored_gain = anchor_aepe - methods["score_top5_oracle_aepe"]["aepe"]
    gates = {
        "capacity_oracle_r4_gain_ge_0_5px": oracle_gain >= 0.5,
        "score_top1_near_ge_2x_random": retrieval["score_top1_oracle_near_percent"] >= 2 * retrieval["random_top1_reference_percent"],
        "score_top5_near_ge_2x_random": retrieval["score_top5_oracle_near_percent"] >= 2 * retrieval["random_top5_reference_percent"],
        "score_top5_candidate_aepe_improves_anchor_by_0_1px": scored_gain >= 0.1,
        "score_top1_does_not_worsen_anchor": methods["score_top1_aepe"]["aepe"] <= anchor_aepe,
    }
    pass_gate = all(gates.values())
    result = {
        "status": "PASS_CANDIDATES_DISCERNIBLE" if pass_gate else "STOP_NO_CANDIDATE_DISCERNIBILITY",
        "last_completed_stage": "P0_700_PAIR_CANDIDATE_GT_JOIN",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_SHA,
        "pairs": 700,
        "metric_scope": "pair-level means over stride-8 source tokens",
        "prediction_manifest": str(manifest),
        "prediction_manifest_sha256": inference_card["manifest_sha256"],
        "prediction_materialized_before_gt_join": True,
        "methods": methods,
        "retrieval": retrieval,
        "margin_entropy_consistency": diagnostics,
        "gates": gates,
        "oracle_r4_gain_px": oracle_gain,
        "score_top5_candidate_gain_px": scored_gain,
        "next_stage": "P1_ALLOWED" if pass_gate else "STOP_BEFORE_P1",
        "ground_truth_policy": "GT first opened only by this join after all 700 candidate bundles were materialized; GT never affected candidate generation, score, rank, or formal output.",
        "backbone_modified": False,
        "checkpoint_modified": False,
        "lucky_r2_started": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(RUN / "P0_RESULT_CARD.json", result)
    write_json(RUN / "RESULT_CARD.json", result)
    (RUN / "RESULT_CARD.md").write_text(
        "# E088A WHU no-GT local refinement P0\n\n"
        f"Status: `{result['status']}`\n\n"
        f"Anchor/oracle-r2/oracle-r4/score-top1/score-top5 AEPE: `{anchor_aepe:.6f}` / "
        f"`{methods['oracle_r2_aepe']['aepe']:.6f}` / `{methods['oracle_r4_aepe']['aepe']:.6f}` / "
        f"`{methods['score_top1_aepe']['aepe']:.6f}` / `{methods['score_top5_oracle_aepe']['aepe']:.6f}`.\n\n"
        f"Score oracle-near top1/top5/top8: `{retrieval['score_top1_oracle_near_percent']:.3f}%` / "
        f"`{retrieval['score_top5_oracle_near_percent']:.3f}%` / `{retrieval['score_top8_oracle_near_percent']:.3f}%`.\n\n"
        f"Gate: `{gates}`. Next stage: `{result['next_stage']}`.\n\n"
        "Frozen E081A-M4 and backbone; prediction-before-GT enforced. No LuckyR2 or local head training was run in P0.\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "inference", "join"), required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.mode == "smoke":
            smoke()
        elif args.mode == "inference":
            inference()
        else:
            join()
    except BaseException as error:
        status = "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "IMPLEMENTATION_FAILED"
        if isinstance(error, torch.cuda.OutOfMemoryError):
            status = "OOM"
        card = {
            "status": status,
            "mode": args.mode,
            "last_completed_stage": "P0_" + args.mode.upper(),
            "cause": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "checkpoint_sha256": EXPECTED_SHA,
            "lucky_r2_started": False,
        }
        write_json(RUN / f"P0_{args.mode.upper()}_FAILURE_RESULT_CARD.json", card)
        (RUN / f"P0_{args.mode.upper()}_FAILURE_RESULT_CARD.md").write_text(
            f"# E088A P0 {args.mode} failure\n\nStatus: `{status}`\n\nCause: `{repr(error)}`\n"
        )
        write_json(RUN / "RESULT_CARD.json", card)
        raise


if __name__ == "__main__":
    main()
