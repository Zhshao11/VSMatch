from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ANCHORS = 128
CANDIDATES = 8
INPUT_DIM = 13
HIDDEN_DIM = 64
RIDGE = 1e-4
MAX_CONDITION = 1e6


@dataclass
class SelectorOutput:
    affine_normalized: torch.Tensor
    candidate_logits: torch.Tensor
    dustbin_logits: torch.Tensor
    candidate_weights: torch.Tensor
    dustbin_weight: torch.Tensor
    residual_xy_normalized: torch.Tensor
    residual_l2_normalized: torch.Tensor
    condition: torch.Tensor
    solver_valid: torch.Tensor
    predicted_grid_error_px: torch.Tensor
    abstain: torch.Tensor
    solver_rounds: int


class CandidateEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, candidate_features):
        return self.network(candidate_features)


def _validate_geometry(source_xy, target_xy, weights):
    if source_xy.ndim != 3 or source_xy.shape[1:] != (ANCHORS, 2):
        raise ValueError(f"source_xy must be [B,128,2], got {tuple(source_xy.shape)}")
    if target_xy.ndim != 4 or target_xy.shape[1:] != (ANCHORS, CANDIDATES, 2):
        raise ValueError(f"target_xy must be [B,128,8,2], got {tuple(target_xy.shape)}")
    if weights.shape != target_xy.shape[:-1]:
        raise ValueError("weights must have shape [B,128,8]")
    if source_xy.shape[0] != target_xy.shape[0]:
        raise ValueError("geometry batch dimensions differ")


def weighted_affine_solve(source_xy, target_xy, weights):
    """FP32 normalized weighted affine with fixed ridge and explicit validity."""
    _validate_geometry(source_xy, target_xy, weights)
    source_xy = source_xy.to(torch.float32)
    target_xy = target_xy.to(torch.float32)
    weights = weights.to(torch.float32)
    batch = source_xy.shape[0]
    design = torch.cat(
        (source_xy, torch.ones(batch, ANCHORS, 1, device=source_xy.device)), dim=-1
    )
    design = design.unsqueeze(2).expand(-1, -1, CANDIDATES, -1).reshape(batch, -1, 3)
    targets = target_xy.reshape(batch, -1, 2)
    flat_weights = weights.reshape(batch, -1)
    finite = (
        torch.isfinite(design).all(dim=(1, 2))
        & torch.isfinite(targets).all(dim=(1, 2))
        & torch.isfinite(flat_weights).all(dim=1)
        & (flat_weights >= 0).all(dim=1)
        & (flat_weights.sum(dim=1) > 1e-6)
    )
    safe_weights = torch.where(finite[:, None], flat_weights, torch.zeros_like(flat_weights))
    weighted_design = safe_weights.unsqueeze(-1) * design
    normal = design.transpose(1, 2) @ weighted_design
    rhs = design.transpose(1, 2) @ (safe_weights.unsqueeze(-1) * targets)
    identity = torch.eye(3, device=normal.device, dtype=torch.float32).expand(batch, -1, -1)
    normal = normal + RIDGE * identity
    normal = torch.where(finite[:, None, None], normal, identity)
    rhs = torch.where(finite[:, None, None], rhs, torch.zeros_like(rhs))
    condition = torch.linalg.cond(normal)
    factor, info = torch.linalg.cholesky_ex(normal, check_errors=False)
    solution = torch.cholesky_solve(rhs, factor).transpose(1, 2)
    valid = (
        finite
        & (info == 0)
        & torch.isfinite(condition)
        & (condition <= MAX_CONDITION)
        & torch.isfinite(solution).all(dim=(1, 2))
    )
    identity_affine = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=solution.device,
        dtype=torch.float32,
    ).expand(batch, -1, -1)
    solution = torch.where(valid[:, None, None], solution, identity_affine)
    return solution, condition, valid


def apply_affine_normalized(affine, source_xy):
    if affine.ndim != 3 or affine.shape[1:] != (2, 3):
        raise ValueError("affine must be [B,2,3]")
    design = torch.cat(
        (source_xy.to(torch.float32), torch.ones_like(source_xy[..., :1])), dim=-1
    )
    return torch.einsum("bij,bnj->bni", affine.to(torch.float32), design)


def affine_flow_at_pixel_points(affine, xy_pixel):
    source_norm = 2.0 * xy_pixel.to(torch.float32) / 511.0 - 1.0
    target_norm = apply_affine_normalized(affine, source_norm)
    target_pixel = (target_norm + 1.0) * 511.0 / 2.0
    return target_pixel - xy_pixel.to(torch.float32)


def dense_affine_flow(affine):
    axis = torch.arange(512, device=affine.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    xy = xy.unsqueeze(0).expand(affine.shape[0], -1, -1)
    return affine_flow_at_pixel_points(affine, xy).reshape(affine.shape[0], 512, 512, 2)


class LearnedGeoSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CandidateEncoder()
        self.unary_head = nn.Linear(HIDDEN_DIM, 1)
        self.dustbin_head = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.residual_update = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 4, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(10, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_inputs(self, candidate_features, source_xy_norm, target_xy_norm):
        if candidate_features.ndim != 4 or candidate_features.shape[1:] != (
            ANCHORS,
            CANDIDATES,
            INPUT_DIM,
        ):
            raise ValueError("candidate_features must be [B,128,8,13]")
        _validate_geometry(
            source_xy_norm,
            target_xy_norm,
            torch.ones_like(target_xy_norm[..., 0]),
        )
        if self.trainable_parameter_count >= 100_000:
            raise RuntimeError("selector parameter budget exceeded")

    def _confidence_features(self, source_xy, weights, dustbin, residual_l2, condition):
        total = weights.sum(dim=(1, 2)).clamp_min(1e-8)
        mean = (weights * residual_l2).sum(dim=(1, 2)) / total
        variance = (weights * (residual_l2 - mean[:, None, None]).square()).sum(dim=(1, 2)) / total
        maximum = (weights * residual_l2).amax(dim=(1, 2))
        effective = (1.0 - dustbin).sum(dim=1) / float(ANCHORS)
        quadrant = torch.stack(
            (
                (source_xy[..., 0] < 0) & (source_xy[..., 1] < 0),
                (source_xy[..., 0] >= 0) & (source_xy[..., 1] < 0),
                (source_xy[..., 0] < 0) & (source_xy[..., 1] >= 0),
                (source_xy[..., 0] >= 0) & (source_xy[..., 1] >= 0),
            ),
            dim=-1,
        ).to(weights.dtype)
        anchor_mass = weights.sum(dim=-1)
        quadrant_mass = (quadrant * anchor_mass.unsqueeze(-1)).sum(dim=1) / total.unsqueeze(-1)
        normalized_condition = torch.log1p(condition.clamp_max(MAX_CONDITION)) / torch.log1p(
            torch.tensor(MAX_CONDITION, device=condition.device)
        )
        return torch.cat(
            (
                dustbin.mean(dim=1, keepdim=True),
                effective.unsqueeze(-1),
                mean.unsqueeze(-1),
                variance.sqrt().unsqueeze(-1),
                maximum.unsqueeze(-1),
                quadrant_mass,
                normalized_condition.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(self, candidate_features, source_xy_norm, target_xy_norm, use_feedback=True):
        self._validate_inputs(candidate_features, source_xy_norm, target_xy_norm)
        embedding = self.encoder(candidate_features)
        unary = self.unary_head(embedding).squeeze(-1)
        pooled = torch.cat((embedding.mean(dim=2), embedding.amax(dim=2)), dim=-1)
        dustbin_logits = self.dustbin_head(pooled).squeeze(-1)
        logits = unary
        rounds = 2 if use_feedback else 1
        for round_index in range(rounds):
            probabilities = F.softmax(torch.cat((logits, dustbin_logits.unsqueeze(-1)), dim=-1), dim=-1)
            weights = probabilities[..., :CANDIDATES]
            dustbin = probabilities[..., CANDIDATES]
            affine, condition, valid = weighted_affine_solve(source_xy_norm, target_xy_norm, weights)
            predicted = apply_affine_normalized(affine, source_xy_norm).unsqueeze(2)
            residual_xy = target_xy_norm.to(torch.float32) - predicted
            residual_l2 = torch.linalg.vector_norm(residual_xy, dim=-1)
            if round_index + 1 < rounds:
                update_input = torch.cat(
                    (embedding, residual_xy, residual_l2.unsqueeze(-1), weights.unsqueeze(-1)), dim=-1
                )
                logits = unary + self.residual_update(update_input).squeeze(-1)
        confidence_features = self._confidence_features(
            source_xy_norm, weights, dustbin, residual_l2, condition
        )
        predicted_error = F.softplus(self.confidence_head(confidence_features).squeeze(-1))
        abstain = (~valid) | (predicted_error > 5.0)
        return SelectorOutput(
            affine_normalized=affine,
            candidate_logits=logits,
            dustbin_logits=dustbin_logits,
            candidate_weights=weights,
            dustbin_weight=dustbin,
            residual_xy_normalized=residual_xy,
            residual_l2_normalized=residual_l2,
            condition=condition,
            solver_valid=valid,
            predicted_grid_error_px=predicted_error,
            abstain=abstain,
            solver_rounds=rounds,
        )


@torch.no_grad()
def nonlearned_soft_em(scores, source_xy_norm, target_xy_norm):
    if scores.shape[1:] != (ANCHORS, CANDIDATES):
        raise ValueError("scores must be [B,128,8]")
    weights = F.softmax(scores.to(torch.float32) / 0.1, dim=-1)
    first, _, valid_first = weighted_affine_solve(source_xy_norm, target_xy_norm, weights)
    predicted = apply_affine_normalized(first, source_xy_norm).unsqueeze(2)
    residual_token = torch.linalg.vector_norm(target_xy_norm - predicted, dim=-1) * 511.0 / 16.0
    robust = -0.5 * torch.minimum((residual_token / 2.0).square(), torch.tensor(9.0, device=scores.device))
    weights = F.softmax(scores.to(torch.float32) / 0.1 + robust, dim=-1)
    second, condition, valid_second = weighted_affine_solve(source_xy_norm, target_xy_norm, weights)
    return second, condition, valid_first & valid_second
