from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.sparsegeo.e003a.selector import affine_flow_at_pixel_points


@dataclass
class LossOutput:
    total: torch.Tensor
    candidate: torch.Tensor
    grid: torch.Tensor
    confidence: torch.Tensor
    actual_grid_error_px: torch.Tensor


def candidate_distribution_loss(candidate_logits, dustbin_logit, target_distribution):
    logits = torch.cat((candidate_logits, dustbin_logit.unsqueeze(-1)), dim=-1)
    if logits.shape != target_distribution.shape or logits.shape[-1] != 9:
        raise ValueError("candidate distribution tensors must have equal [B,128,9] shape")
    if not torch.allclose(target_distribution.sum(dim=-1), torch.ones_like(target_distribution[..., 0])):
        raise ValueError("candidate target distributions must sum to one")
    return -(target_distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def affine_grid_loss(affine, grid_xy_pixel, grid_truth_flow):
    predicted_flow = affine_flow_at_pixel_points(affine, grid_xy_pixel)
    residual = predicted_flow - grid_truth_flow.to(torch.float32)
    endpoint = torch.linalg.vector_norm(residual, dim=-1)
    charbonnier = torch.sqrt((endpoint / 8.0).square() + 1e-3**2).mean()
    return charbonnier, endpoint.mean(dim=-1)


def confidence_calibration_loss(predicted_grid_error_px, actual_grid_error_px):
    return F.smooth_l1_loss(
        torch.log1p(predicted_grid_error_px),
        torch.log1p(actual_grid_error_px.detach()),
    )


def selector_losses(output, target_distribution, grid_xy_pixel, grid_truth_flow):
    candidate = candidate_distribution_loss(
        output.candidate_logits, output.dustbin_logits, target_distribution
    )
    grid, actual = affine_grid_loss(output.affine_normalized, grid_xy_pixel, grid_truth_flow)
    confidence = confidence_calibration_loss(output.predicted_grid_error_px, actual)
    total = candidate + grid + 0.2 * confidence
    return LossOutput(
        total=total,
        candidate=candidate,
        grid=grid,
        confidence=confidence,
        actual_grid_error_px=actual,
    )
