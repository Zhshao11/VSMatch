#!/usr/bin/env python3
"""LuckyR1 v2 evidence modules.

This module keeps the validated B relation operator from v1 and replaces the
single-expectation geometric belief with sparse, absolute-coordinate modes.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

import luckyr1_models as v1

IMPLEMENTATION_VERSION = "luckyr1-bc-v2-sparse-multimode-20260913"

# Re-export the unchanged model and path helpers used by the training pipeline.
BCModel = v1.BCModel
M4Model = v1.M4Model
F2_CKPT = v1.F2_CKPT
F3_CKPT = v1.F3_CKPT
lsr = v1.lsr


def _shift_neighbor(value: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a cardinal source neighbor without wrapping image boundaries."""
    shifted = torch.roll(value, shifts=(-dy, -dx), dims=(1, 2))
    valid = torch.ones(value.shape[:3], device=value.device, dtype=value.dtype)
    if dx > 0:
        valid[:, :, -dx:] = 0
    elif dx < 0:
        valid[:, :, :-dx] = 0
    if dy > 0:
        valid[:, -dy:, :] = 0
    elif dy < 0:
        valid[:, :-dy, :] = 0
    return shifted, valid


class SparseGeometryBelief(nn.Module):
    """Candidate-to-candidate geometry messages with recurrent sparse modes.

    Mode indices are selected with hard top-k (and are therefore not
    differentiable); selected probabilities, relation logits, and all evidence
    strengths remain differentiable. Recurrent modes are detached between GRU
    iterations to bound memory and to avoid treating repeated queries as new
    independent observations.
    """

    def __init__(self, neighbor_modes: int = 2, recurrent_modes: int = 4):
        super().__init__()
        self.neighbor_modes = neighbor_modes
        self.recurrent_modes = recurrent_modes
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.unmatched_logits = nn.Parameter(torch.full((4,), -1.0))
        self.geometry_strength = nn.Parameter(torch.tensor(-5.0))
        self.temporal_strength = nn.Parameter(torch.tensor(-5.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.6931471805599453))

    @staticmethod
    def _top_modes(probability: torch.Tensor, candidates: torch.Tensor, count: int):
        # probability: Bx81xHxW; candidates: BxHxWx81x2
        values, indices = torch.topk(probability.permute(0, 2, 3, 1), k=count, dim=-1)
        targets = torch.gather(candidates, 3, indices[..., None].expand(*indices.shape, 2))
        return values, targets

    def _spatial_message(
        self,
        candidates: torch.Tensor,
        mode_probability: torch.Tensor,
        mode_targets: torch.Tensor,
        jacobian: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        message = torch.zeros(candidates.shape[:-1], device=candidates.device, dtype=candidates.dtype)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_targets, valid = _shift_neighbor(mode_targets, dx, dy)
            neighbor_probability, _ = _shift_neighbor(mode_probability, dx, dy)
            direction = candidates.new_tensor([float(dx), float(dy)])
            transformed = torch.einsum("bchwk,k->bhwc", jacobian, direction)
            predicted_neighbor = candidates + transformed[..., None, :]
            residual2 = (
                neighbor_targets[..., None, :, :] - predicted_neighbor[..., :, None, :]
            ).square().sum(dim=-1)
            compatibility = torch.exp(-residual2.clamp(max=16.0 * sigma2) / sigma2)
            message = message + valid[..., None] * (
                compatibility * neighbor_probability[..., None, :]
            ).sum(dim=-1)
        return message

    def forward(self, corr, coords0, coords1, previous_modes=None):
        batch, channels, height, width = corr.shape
        if channels != 4 * 81:
            raise ValueError(f"expected four radius-4 CorrBlock levels, got {channels} channels")
        logits = corr.reshape(batch, 4, 81, height, width)
        offsets = v1._offsets(corr.device, corr.dtype)
        jacobian = v1._jacobian(coords1)
        sigma2 = self.log_sigma.exp().square().clamp_min(0.25)
        level_weight = torch.softmax(self.level_logits, dim=0)
        enhanced_levels = []
        candidates_by_level = []
        unmatched_mass = []

        for level in range(4):
            unmatched = self.unmatched_logits[level].expand(batch, 1, height, width)
            probability = torch.softmax(torch.cat([logits[:, level], unmatched], dim=1), dim=1)
            matched = probability[:, :81]
            unmatched_mass.append(probability[:, 81])
            candidates = (
                coords1.permute(0, 2, 3, 1)[..., None, :]
                + offsets[None, None, None] * (2**level)
            )
            candidates_by_level.append(candidates)
            mode_probability, mode_targets = self._top_modes(
                matched, candidates, self.neighbor_modes
            )
            message = self._spatial_message(
                candidates, mode_probability, mode_targets, jacobian, sigma2
            )
            prior = message - message.mean(dim=-1, keepdim=True)

            if previous_modes is not None:
                residual2 = (
                    candidates[..., :, None, :] - previous_modes["targets"][..., None, :, :]
                ).square().sum(dim=-1)
                temporal = (
                    torch.exp(-residual2.clamp(max=16.0 * sigma2) / sigma2)
                    * previous_modes["probabilities"][..., None, :]
                ).sum(dim=-1)
                prior = prior + torch.sigmoid(self.temporal_strength) * (
                    temporal - temporal.mean(dim=-1, keepdim=True)
                )

            level_scale = 4.0 * level_weight[level]
            enhanced = logits[:, level] + (
                torch.sigmoid(self.geometry_strength) * level_scale * prior.permute(0, 3, 1, 2)
            )
            enhanced_levels.append(enhanced)

        # Persist several absolute target modes, not a global coordinate mean.
        state_probabilities, state_targets = [], []
        for level, enhanced in enumerate(enhanced_levels):
            unmatched = self.unmatched_logits[level].expand(batch, 1, height, width)
            probability = torch.softmax(torch.cat([enhanced, unmatched], dim=1), dim=1)[:, :81]
            values, targets = self._top_modes(
                probability, candidates_by_level[level], self.neighbor_modes
            )
            state_probabilities.append(values * level_weight[level])
            state_targets.append(targets)
        all_probability = torch.cat(state_probabilities, dim=-1)
        all_targets = torch.cat(state_targets, dim=3)
        state_probability, state_index = torch.topk(
            all_probability, k=min(self.recurrent_modes, all_probability.shape[-1]), dim=-1
        )
        state_targets = torch.gather(
            all_targets, 3, state_index[..., None].expand(*state_index.shape, 2)
        )
        next_modes = {
            "targets": state_targets.detach(),
            "probabilities": state_probability.detach(),
        }
        diagnostics = {
            "matched_mass": (1.0 - torch.stack(unmatched_mass, dim=0)).mean().detach(),
            "belief_sigma": self.log_sigma.exp().detach(),
            "recurrent_modes": self.recurrent_modes,
            "hard_mode_selection": True,
        }
        return torch.cat(enhanced_levels, dim=1), next_modes, diagnostics


class EvidenceAdapter(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.belief = SparseGeometryBelief() if kind in ("A", "C") else None
        self.relation = v1.AlignedRelation() if kind in ("B", "C") else None

    def forward(self, corr, source_feature, target_feature, coords0, coords1, previous_modes):
        diagnostics = {}
        if self.relation is not None:
            corr = self.relation(corr, source_feature, target_feature, coords0, coords1)
        if self.belief is not None:
            corr, previous_modes, diagnostics = self.belief(
                corr, coords0, coords1, previous_modes
            )
        return corr, previous_modes, diagnostics


class LuckySafeE073(v1.LuckySafeE073):
    def __init__(self, e072, cfg, variant: str, device, lucky_kind: str):
        super().__init__(e072, cfg, variant, device, lucky_kind)
        self.evidence = EvidenceAdapter(lucky_kind)


class HistoricalLuckyE073(v1.HistoricalLuckyE073):
    def __init__(self, base: nn.Module, lucky_kind: str):
        super().__init__(base, lucky_kind)
        self.evidence = EvidenceAdapter(lucky_kind)


class HistoricalLuckyM4(nn.Module):
    """Protected E075A-M4 with trainable v2 Lucky evidence only."""

    def __init__(self, checkpoint: Path, expected_sha256: str, lucky_kind: str, device,
                 verify_checksum: bool = True, cache_paths: dict[str, str] | None = None):
        super().__init__()
        sys.path.insert(0, str(v1.TMP))
        import e075a_preserve_high_precision_recover_c2c5 as e075

        e074, _e072, e073 = e075.add_paths()
        if cache_paths:
            e075.B_CKPT = Path(cache_paths["b"])
            e075.C_CKPT = Path(cache_paths["c"])
            e074.F2_CKPT = Path(cache_paths["e073_f2"])
            e074.F3_CKPT = Path(cache_paths["e073_f3"])
            e073.V1_60K = Path(cache_paths["e072_v1"])
            e073.V2_60K = Path(cache_paths["e072_v2"])
            _e072.E046_CKPT = Path(cache_paths["e046"])
        if verify_checksum:
            hasher = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != expected_sha256:
                raise RuntimeError(f"protected M4 checksum mismatch: {digest}")
        model = e075.PairFusion(e074, "region", device).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"], strict=True)
        model.b.f2 = HistoricalLuckyE073(model.b.f2, lucky_kind)
        model.b.f3 = HistoricalLuckyE073(model.b.f3, lucky_kind)
        model.c.student = HistoricalLuckyE073(model.c.student, lucky_kind)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, EvidenceAdapter):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.model = model
        self.e074 = e074

    def forward(self, source, target, iters=12):
        return self.model(source, target, iters=iters, lsr_fn=self.e074.lsr)
