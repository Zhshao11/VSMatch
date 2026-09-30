"""Dependency-isolated copy of the frozen full-image affine metric equations."""

import numpy as np
import torch


def affine_flow_at_pixel_points(affine, xy_pixel):
    if affine.ndim != 3 or affine.shape[1:] != (2, 3):
        raise ValueError("affine must be [B,2,3]")
    source_norm = 2.0 * xy_pixel.to(torch.float32) / 511.0 - 1.0
    design = torch.cat((source_norm, torch.ones_like(source_norm[..., :1])), dim=-1)
    target_norm = torch.einsum("bij,bnj->bni", affine.to(torch.float32), design)
    target_pixel = (target_norm + 1.0) * 511.0 / 2.0
    return target_pixel - xy_pixel.to(torch.float32)


def dense_affine_flow(affine):
    axis = torch.arange(512, device=affine.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    xy = xy.unsqueeze(0).expand(affine.shape[0], -1, -1)
    return affine_flow_at_pixel_points(affine, xy).reshape(affine.shape[0], 512, 512, 2)


def pair_metrics(affine, flow_truth, grid_xy, grid_truth):
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


def summarize(values):
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
