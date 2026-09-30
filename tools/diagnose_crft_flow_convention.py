#!/usr/bin/env python3
"""Compare CRFT predictions against plausible affine-flow conventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--os-variant", default="512_15Trans0.1Scale30Rot")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def affine_flow(matrix: np.ndarray, height: int, width: int) -> torch.Tensor:
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    target_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]
    target_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]
    flow = np.stack((target_x - grid_x, target_y - grid_y), axis=0).astype(np.float32)
    return torch.from_numpy(flow)


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    source_height, source_width = flow.shape[-2:]
    target_height, target_width = size
    resized = F.interpolate(
        flow[None], size=size, mode="bilinear", align_corners=True
    )[0]
    resized[0] *= target_width / source_width
    resized[1] *= target_height / source_height
    return resized


def endpoint_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target, dim=0).mean())


def fit_affine(flow: torch.Tensor) -> tuple[np.ndarray, float]:
    """Least-squares fit a source-to-target affine matrix to a dense flow."""
    flow_np = flow.detach().cpu().numpy().astype(np.float64)
    height, width = flow_np.shape[-2:]
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    design = np.stack((grid_x.ravel(), grid_y.ravel(), np.ones(height * width)), axis=1)
    target = np.stack(
        ((grid_x + flow_np[0]).ravel(), (grid_y + flow_np[1]).ravel()),
        axis=1,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    matrix = coefficients.T
    fitted = affine_flow(matrix, height, width)
    return matrix, endpoint_error(flow, fitted)


def flow_scale_fit(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Return best scalar for prediction->target and their flattened cosine similarity."""
    prediction_flat = prediction.double().reshape(-1)
    target_flat = target.double().reshape(-1)
    denominator = torch.dot(prediction_flat, prediction_flat)
    scale = torch.dot(prediction_flat, target_flat) / denominator.clamp_min(1e-12)
    cosine = torch.dot(prediction_flat, target_flat) / (
        prediction_flat.norm() * target_flat.norm()
    ).clamp_min(1e-12)
    return float(scale), float(cosine)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def binned_epe(records: list[dict[str, float]], field: str, edges: list[float]) -> list[dict[str, float]]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = [r["epe"] for r in records if low <= r[field] < high]
        if selected:
            rows.append({
                "low": low,
                "high": high,
                "count": len(selected),
                "mean_epe": float(np.mean(selected)),
                "median_epe": float(np.median(selected)),
            })
    return rows


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))

    from src.config.default import get_cfg_defaults
    from src.datasets.osdataset import OSDataset
    from src.lightning.lightning_crft import PL_CRFT

    config = get_cfg_defaults()
    config.merge_from_file(str(repo / "configs/crft/outdoor/visible_thermal.py"))
    config.merge_from_file(str(repo / "configs/data/osdataset_640.py"))
    config.DATASET.OS_VARIANT = args.os_variant

    dataset = OSDataset(
        root_dir=str(args.data_root.expanduser().resolve()),
        list_path=str(args.data_root.expanduser().resolve() / "test_list.txt"),
        mode="test",
        img_resize=config.DATASET.OS_IMG_RESIZE,
        df=config.DATASET.OS_DF,
        img_padding=config.DATASET.OS_IMG_PAD,
        os_variant=args.os_variant,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    module = PL_CRFT(
        config,
        pretrained_ckpt=args.checkpoint,
        require_full_checkpoint=True,
    ).cuda().eval()

    manifest = {
        record["pair_id"]: record
        for record in map(json.loads, args.manifest.read_text(encoding="utf-8").splitlines())
    }
    errors: dict[str, list[float]] = {
        "released_flow": [],
        "forward_affine": [],
        "inverse_affine": [],
        "negative_forward": [],
        "negative_inverse": [],
        "swapped_forward": [],
    }
    flow_file_residuals: list[float] = []
    affine_projection_residuals: list[float] = []
    target_affine_projection_residuals: list[float] = []
    flow_scale_fits: list[float] = []
    flow_cosines: list[float] = []
    parameter_errors: dict[str, list[float]] = {
        "linear_frobenius": [],
        "translation_l2": [],
    }
    sample_records: list[dict[str, float | str]] = []

    with torch.inference_mode():
        for index, batch in enumerate(dataloader):
            if index >= args.max_samples:
                break
            pair_stem = Path(batch["pair_names"][0][0]).stem
            pair_id = pair_stem[:-4] if pair_stem.endswith("_opt") else pair_stem
            record = manifest[pair_id]
            affine = np.asarray(record["affine_source_to_warped"], dtype=np.float64)
            affine_3x3 = np.vstack((affine, (0.0, 0.0, 1.0)))
            inverse = np.linalg.inv(affine_3x3)[:2]

            full_forward = affine_flow(affine, 512, 512)
            full_inverse = affine_flow(inverse, 512, 512)
            target_size = tuple(batch["flow"].shape[-2:])
            candidates = {
                "forward_affine": resize_flow(full_forward, target_size),
                "inverse_affine": resize_flow(full_inverse, target_size),
            }
            candidates["negative_forward"] = -candidates["forward_affine"]
            candidates["negative_inverse"] = -candidates["inverse_affine"]
            candidates["swapped_forward"] = candidates["forward_affine"][[1, 0]]

            released_flow = batch["flow"][0]
            flow_file_residuals.append(endpoint_error(released_flow, candidates["forward_affine"]))
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.cuda(non_blocking=True)
            module.matcher(batch)
            prediction = batch["flow_f_full"][0].detach().cpu()
            sample_epe = endpoint_error(prediction, released_flow)
            errors["released_flow"].append(sample_epe)
            for name, target in candidates.items():
                errors[name].append(endpoint_error(prediction, target))

            prediction_affine, prediction_residual = fit_affine(prediction)
            target_affine, target_residual = fit_affine(released_flow)
            affine_projection_residuals.append(prediction_residual)
            target_affine_projection_residuals.append(target_residual)
            parameter_errors["linear_frobenius"].append(
                float(np.linalg.norm(prediction_affine[:, :2] - target_affine[:, :2]))
            )
            parameter_errors["translation_l2"].append(
                float(np.linalg.norm(prediction_affine[:, 2] - target_affine[:, 2]))
            )
            scale_fit, cosine = flow_scale_fit(prediction, released_flow)
            flow_scale_fits.append(scale_fit)
            flow_cosines.append(cosine)
            sample_records.append({
                "pair_id": pair_id,
                "epe": sample_epe,
                "abs_rotation_deg": abs(float(record["rotation_deg"])),
                "abs_scale_delta": abs(float(record["scale"]) - 1.0),
                "translation_l2": float(np.linalg.norm(record["translation_px"])),
                "prediction_affine_residual": prediction_residual,
                "best_flow_scale": scale_fit,
                "flow_cosine": cosine,
            })

    report = {
        "sample_count": len(errors["released_flow"]),
        "mean_epe": {name: float(np.mean(values)) for name, values in errors.items()},
        "median_epe": {name: float(np.median(values)) for name, values in errors.items()},
        "max_generated_vs_loaded_flow_epe": float(max(flow_file_residuals)),
        "prediction_affine_projection_residual": summarize(affine_projection_residuals),
        "target_affine_projection_residual": summarize(target_affine_projection_residuals),
        "best_prediction_to_target_flow_scale": summarize(flow_scale_fits),
        "prediction_target_flow_cosine": summarize(flow_cosines),
        "fitted_affine_parameter_error": {
            name: summarize(values) for name, values in parameter_errors.items()
        },
        "epe_by_abs_rotation_deg": binned_epe(
            sample_records, "abs_rotation_deg", [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 46.0]
        ),
        "epe_by_abs_scale_delta": binned_epe(
            sample_records, "abs_scale_delta", [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.21]
        ),
        "epe_by_translation_l2": binned_epe(
            sample_records, "translation_l2", [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
        ),
        "worst_samples": sorted(sample_records, key=lambda row: row["epe"], reverse=True)[:20],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
