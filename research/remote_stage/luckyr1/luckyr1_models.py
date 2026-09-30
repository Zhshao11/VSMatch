#!/usr/bin/env python3
"""Model-facing clean-lineage B/C, M4, and LuckyR1 modules."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/mnt/sda/zhushihao/VS_Match")
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
PARENT = ROOT / "GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN"
F2_CKPT = PARENT / "foundation_safe_v1v1/F2/checkpoints/F2_step120000.pt"
F3_CKPT = PARENT / "foundation_safe_v1v1/F3/checkpoints/F3_step120000.pt"


def imports():
    sys.path.insert(0, str(TMP))
    import e072a_layerwise_frontend_alignment as e072

    e072.add_paths()
    from lucky_safe_e073_foundation import SafeE073
    return e072, SafeE073


def lsr(flow):
    from core.LSRnet.LSmodel import LSR

    return LSR(1, 512, 512, flow, 200000).to(flow.device)


def load_safe(variant: str, checkpoint: Path, device, lucky_kind: str | None = None):
    e072, SafeE073 = imports()
    cls = LuckySafeE073 if lucky_kind else SafeE073
    model = cls(e072, e072.cfg(), variant, device, lucky_kind) if lucky_kind else cls(e072, e072.cfg(), variant, device)
    payload = torch.load(checkpoint, map_location=device)["model"]
    result = model.base.load_state_dict(payload, strict=True) if lucky_kind else model.load_state_dict(payload, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict foundation load failed: {result}")
    return model


class FlowGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 1, 1))

    def forward(self, value):
        return torch.sigmoid(self.net(value))


class BCModel(nn.Module):
    """Source-equivalent E074 B gate or C distilled student on split-safe foundations."""

    def __init__(self, variant: str, device, f2_checkpoint=F2_CKPT, f3_checkpoint=F3_CKPT, lucky_kind: str | None = None):
        super().__init__()
        self.variant = variant.upper()
        branch_kind = lucky_kind if self.variant == "B" else None
        self.f2 = load_safe("F2", Path(f2_checkpoint), device, branch_kind)
        self.f3 = load_safe("F3", Path(f3_checkpoint), device, branch_kind)
        for parameter in list(self.f2.parameters()) + list(self.f3.parameters()):
            parameter.requires_grad_(False)
        if self.variant == "B":
            self.gate = FlowGate(9)
            if lucky_kind:
                for module in (self.f2.evidence, self.f3.evidence):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        elif self.variant == "C":
            self.student = load_safe("F3", Path(f3_checkpoint), device, lucky_kind)
        else:
            raise ValueError(self.variant)

    @staticmethod
    def gate_input(flow2, flow3):
        difference = flow2 - flow3
        post2, post3 = lsr(flow2).detach(), lsr(flow3).detach()
        return torch.cat([flow2.detach(), flow3.detach(), difference.detach(), torch.linalg.vector_norm(difference.detach(), dim=1, keepdim=True), torch.linalg.vector_norm(flow2.detach() - post2, dim=1, keepdim=True), torch.linalg.vector_norm(flow3.detach() - post3, dim=1, keepdim=True)], dim=1)

    def forward(self, source, target, iters=12):
        if self.variant == "B":
            flow2, flow3 = self.f2(source, target, iters), self.f3(source, target, iters)
            outputs = []
            for value2, value3 in zip(flow2, flow3):
                gate = self.gate(self.gate_input(value2, value3))
                outputs.append((1 - gate) * value3 + gate * value2)
            return outputs
        with torch.no_grad():
            flow2, flow3 = self.f2(source, target, iters), self.f3(source, target, iters)
        outputs = self.student(source, target, iters)
        self.teacher_average = [0.5 * (value2 + value3) for value2, value3 in zip(flow2, flow3)]
        return outputs


class M4Model(nn.Module):
    def __init__(self, device, b_checkpoint: Path, c_checkpoint: Path, lucky_kind: str | None = None):
        super().__init__()
        self.b = BCModel("B", device, lucky_kind=lucky_kind)
        self.c = BCModel("C", device, lucky_kind=lucky_kind)
        b_state = torch.load(b_checkpoint, map_location=device)["model"]
        c_state = torch.load(c_checkpoint, map_location=device)["model"]
        if lucky_kind:
            raw_b, raw_c = BCModel("B", device), BCModel("C", device)
            raw_b.load_state_dict(b_state, strict=True)
            raw_c.load_state_dict(c_state, strict=True)
            self.b.f2.base.load_state_dict(raw_b.f2.state_dict(), strict=True)
            self.b.f3.base.load_state_dict(raw_b.f3.state_dict(), strict=True)
            self.b.gate.load_state_dict(raw_b.gate.state_dict(), strict=True)
            self.c.f2.load_state_dict(raw_c.f2.state_dict(), strict=True)
            self.c.f3.load_state_dict(raw_c.f3.state_dict(), strict=True)
            self.c.student.base.load_state_dict(raw_c.student.state_dict(), strict=True)
            del raw_b, raw_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self.b.load_state_dict(b_state, strict=True)
            self.c.load_state_dict(c_state, strict=True)
        for parameter in list(self.b.parameters()) + list(self.c.parameters()):
            parameter.requires_grad_(False)
        self.gate = FlowGate(16)

    @staticmethod
    def gate_input(flow_b, flow_c):
        post_b, post_c = lsr(flow_b).detach(), lsr(flow_c).detach()
        return torch.cat([flow_b.detach(), flow_c.detach(), (flow_b - flow_c).detach(), torch.abs(flow_b - flow_c).detach(), post_b, post_c, torch.abs(post_b - flow_b).detach(), torch.abs(post_c - flow_c).detach()], dim=1)

    def forward(self, source, target, iters=12):
        flow_b, flow_c = self.b(source, target, iters), self.c(source, target, iters)
        outputs = []
        for value_b, value_c in zip(flow_b, flow_c):
            gate = self.gate(self.gate_input(value_b, value_c))
            gate = F.avg_pool2d(gate, 4, 4)
            gate = F.interpolate(gate, value_b.shape[-2:], mode="bilinear", align_corners=False)
            outputs.append((1 - gate) * value_b + gate * value_c)
        return outputs


def _offsets(device, dtype):
    axis = torch.linspace(-4, 4, 9, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(81, 2)


def _jacobian(coords):
    dx = F.pad(coords[:, :, :, 2:] - coords[:, :, :, :-2], (1, 1, 0, 0), mode="replicate") * 0.5
    dy = F.pad(coords[:, :, 2:, :] - coords[:, :, :-2, :], (0, 0, 1, 1), mode="replicate") * 0.5
    return torch.stack([dx, dy], dim=-1)


def _sample(feature, coordinates):
    batch, _channels, height, width = feature.shape
    shape = coordinates.shape
    grid = coordinates.reshape(batch, shape[1], -1, 2)
    normalizer = torch.tensor([max(width - 1, 1), max(height - 1, 1)], device=feature.device, dtype=feature.dtype)
    grid = 2.0 * grid / normalizer - 1.0
    sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.reshape(batch, feature.shape[1], *shape[1:-1])


class GeometryBelief(nn.Module):
    """Explicit candidate belief with unmatched mass and absolute-coordinate recurrence."""

    def __init__(self):
        super().__init__()
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.unmatched_logits = nn.Parameter(torch.full((4,), -1.0))
        self.geometry_strength = nn.Parameter(torch.tensor(-5.0))
        self.temporal_strength = nn.Parameter(torch.tensor(-5.0))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, corr, coords0, coords1, previous_target=None):
        batch, _channels, height, width = corr.shape
        logits = corr.reshape(batch, 4, 81, height, width)
        offsets = _offsets(corr.device, corr.dtype)
        levels, expected = [], []
        matched_mass = []
        for level in range(4):
            unmatched = self.unmatched_logits[level].expand(batch, 1, height, width)
            probability = torch.softmax(torch.cat([logits[:, level], unmatched], dim=1), dim=1)
            matched = probability[:, :81]
            candidates = coords1.permute(0, 2, 3, 1)[:, :, :, None, :] + offsets[None, None, None] * (2**level)
            mass = matched.sum(dim=1).clamp_min(1e-6)
            target = (matched.permute(0, 2, 3, 1)[..., None] * candidates).sum(dim=3) / mass[..., None]
            levels.append(candidates)
            expected.append(target)
            matched_mass.append(mass)
        level_weight = torch.softmax(self.level_logits, dim=0)
        absolute_target = sum(level_weight[i] * expected[i] for i in range(4))
        expected_flow = absolute_target.permute(0, 3, 1, 2) - coords0
        geometry_target = coords0 + F.avg_pool2d(expected_flow, 3, 1, 1)
        sigma2 = self.log_sigma.exp().square().clamp_min(0.25)
        enhanced = []
        for level in range(4):
            candidates = levels[level]
            geometric = -((candidates - geometry_target.permute(0, 2, 3, 1)[..., None, :]).square().sum(dim=-1) / sigma2)
            prior = geometric.permute(0, 3, 1, 2)
            if previous_target is not None:
                temporal = -((candidates - previous_target.permute(0, 2, 3, 1)[..., None, :]).square().sum(dim=-1) / sigma2)
                prior = prior + torch.sigmoid(self.temporal_strength) * temporal.permute(0, 3, 1, 2)
            prior = prior - prior.mean(dim=1, keepdim=True)
            enhanced.append(logits[:, level] + torch.sigmoid(self.geometry_strength) * prior)
        diagnostics = {"matched_mass": torch.stack(matched_mass).mean().detach(), "belief_sigma": self.log_sigma.exp().detach()}
        return torch.cat(enhanced, dim=1), absolute_target.permute(0, 3, 1, 2).detach(), diagnostics


class AlignedRelation(nn.Module):
    """Compare within-modality relations after warping target offsets by the current Jacobian."""

    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(-5.0))
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, corr, source_feature, target_feature, coords0, coords1):
        batch, _channels, height, width = corr.shape
        offsets = _offsets(corr.device, corr.dtype)
        source_xy = coords0.permute(0, 2, 3, 1)
        directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=corr.device, dtype=corr.dtype)
        source_center = F.normalize(source_feature.float(), dim=1)
        source_relation = []
        for direction in directions:
            neighbor = F.normalize(_sample(source_feature.float(), source_xy + direction), dim=1)
            source_relation.append((source_center * neighbor).sum(dim=1))
        source_relation = torch.stack(source_relation, dim=-1)[..., None, :]
        jacobian = _jacobian(coords1)
        raw = corr.reshape(batch, 4, 81, height, width)
        relation_levels = []
        for level in range(4):
            feature = target_feature.float()
            for _ in range(level):
                feature = F.avg_pool2d(feature, 2, 2)
            candidates = coords1.permute(0, 2, 3, 1)[:, :, :, None, :] / (2**level) + offsets[None, None, None]
            target_center = F.normalize(_sample(feature, candidates), dim=1)
            target_relations = []
            for direction in directions:
                transformed = torch.einsum("bchwk,k->bchw", jacobian, direction).permute(0, 2, 3, 1) / (2**level)
                neighbor = F.normalize(_sample(feature, candidates + transformed[..., None, :]), dim=1)
                target_relations.append((target_center * neighbor).sum(dim=1))
            target_relation = torch.stack(target_relations, dim=-1)
            evidence = -self.scale.exp().clamp(max=20.0) * (target_relation - source_relation).square().mean(dim=-1)
            evidence = evidence.permute(0, 3, 1, 2)
            evidence = evidence - evidence.mean(dim=1, keepdim=True)
            relation_levels.append(raw[:, level] + torch.sigmoid(self.strength) * evidence)
        return torch.cat(relation_levels, dim=1)


class EvidenceAdapter(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.belief = GeometryBelief() if kind in ("A", "C") else None
        self.relation = AlignedRelation() if kind in ("B", "C") else None

    def forward(self, corr, source_feature, target_feature, coords0, coords1, previous_target):
        diagnostics = {}
        if self.relation is not None:
            corr = self.relation(corr, source_feature, target_feature, coords0, coords1)
        if self.belief is not None:
            corr, previous_target, diagnostics = self.belief(corr, coords0, coords1, previous_target)
        return corr, previous_target, diagnostics


class LuckySafeE073(nn.Module):
    """SafeE073-compatible matcher whose real CorrBlock output is adapted before BasicUpdateBlock."""

    def __init__(self, e072, cfg, variant: str, device, lucky_kind: str):
        super().__init__()
        _, SafeE073 = imports()
        self.base = SafeE073(e072, cfg, variant, device)
        self.variant = variant
        self.evidence = EvidenceAdapter(lucky_kind)

    def forward(self, source, target, iters=12):
        from core.corr import CorrBlock

        image1, image2 = 2 * (source / 255.0) - 1.0, 2 * (target / 255.0) - 1.0
        f1s, f1t, f2s, f2t = self.base.features(source, target, image1, image2)
        if self.variant == "F2":
            gate_s = torch.sigmoid(self.base.fuse_gate(torch.cat([f1s, f2s], dim=1)))
            gate_t = torch.sigmoid(self.base.fuse_gate(torch.cat([f1t, f2t], dim=1)))
            feature_s = (1 - gate_s) * f1s + gate_s * self.base.fuse_proj(torch.cat([f1s, f2s], dim=1))
            feature_t = (1 - gate_t) * f1t + gate_t * self.base.fuse_proj(torch.cat([f1t, f2t], dim=1))
            corr_fn = CorrBlock(feature_s.float(), feature_t.float(), radius=4)
        else:
            alpha = torch.sigmoid(self.base.corr_gate_logit)
            corr1, corr2 = CorrBlock(f1s.float(), f1t.float(), radius=4), CorrBlock(f2s.float(), f2t.float(), radius=4)
            corr_fn = lambda coordinates: (1 - alpha) * corr1(coordinates) + alpha * corr2(coordinates)
            feature_s, feature_t = (1 - alpha) * f1s + alpha * f2s, (1 - alpha) * f1t + alpha * f2t
        context = self.base.context(image1)
        net, inp = torch.split(context, [128, 128], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)
        coords0, coords1 = self.base.initialize_flow(image1)
        outputs, previous_target = [], None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr, previous_target, diagnostics = self.evidence(corr_fn(coords1), feature_s, feature_t, coords0, coords1, previous_target)
            net, mask, delta = self.base.update(net, inp, corr, coords1 - coords0)
            coords1 = coords1 + delta
            outputs.append(self.base.upsample_flow(coords1 - coords0, mask))
        self.last_evidence = diagnostics
        return outputs


class HistoricalLuckyE073(nn.Module):
    """Inject Lucky evidence into an already-loaded historical E073 matcher."""

    def __init__(self, base: nn.Module, lucky_kind: str):
        super().__init__()
        self.base = base
        self.evidence = EvidenceAdapter(lucky_kind)

    def forward(self, source, target, iters=12, return_features=False):
        from core.corr import CorrBlock

        image1, image2 = 2 * (source / 255.0) - 1.0, 2 * (target / 255.0) - 1.0
        f1s, f1t, f2s, f2t = self.base.branch_features(source, target, image1, image2)
        if self.base.variant == "F2":
            gate_s = torch.sigmoid(self.base.fuse_gate(torch.cat([f1s, f2s], dim=1)))
            gate_t = torch.sigmoid(self.base.fuse_gate(torch.cat([f1t, f2t], dim=1)))
            feature_s = (1 - gate_s) * f1s + gate_s * self.base.fuse_proj(torch.cat([f1s, f2s], dim=1))
            feature_t = (1 - gate_t) * f1t + gate_t * self.base.fuse_proj(torch.cat([f1t, f2t], dim=1))
            corr_fn = CorrBlock(feature_s.float(), feature_t.float(), radius=self.base.corr_radius)
        elif self.base.variant == "F3":
            alpha = torch.sigmoid(self.base.corr_gate_logit)
            corr1, corr2 = CorrBlock(f1s.float(), f1t.float(), radius=self.base.corr_radius), CorrBlock(f2s.float(), f2t.float(), radius=self.base.corr_radius)
            corr_fn = lambda coordinates: (1 - alpha) * corr1(coordinates) + alpha * corr2(coordinates)
            feature_s, feature_t = (1 - alpha) * f1s + alpha * f2s, (1 - alpha) * f1t + alpha * f2t
        else:
            raise ValueError(f"LuckyR1 only supports E073 F2/F3, got {self.base.variant}")
        context = self.base.context(image1)
        net, inp = torch.split(context, [128, 128], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)
        coords0, coords1 = self.base.initialize_flow(image1)
        outputs, previous_target = [], None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr, previous_target, diagnostics = self.evidence(corr_fn(coords1), feature_s, feature_t, coords0, coords1, previous_target)
            net, mask, delta = self.base.update(net, inp, corr, coords1 - coords0)
            coords1 = coords1 + delta
            outputs.append(self.base.upsample_flow(coords1 - coords0, mask))
        self.last_evidence = diagnostics
        if return_features:
            return outputs, feature_s, feature_t, diagnostics
        return outputs


class HistoricalLuckyM4(nn.Module):
    """Protected E075A-M4 with trainable Lucky evidence only."""

    def __init__(self, checkpoint: Path, expected_sha256: str, lucky_kind: str, device, verify_checksum: bool = True, cache_paths: dict[str, str] | None = None):
        super().__init__()
        import hashlib

        sys.path.insert(0, str(TMP))
        import e075a_preserve_high_precision_recover_c2c5 as e075

        e074, e072, e073 = e075.add_paths()
        if cache_paths:
            e075.B_CKPT = Path(cache_paths["b"])
            e075.C_CKPT = Path(cache_paths["c"])
            e074.F2_CKPT = Path(cache_paths["e073_f2"])
            e074.F3_CKPT = Path(cache_paths["e073_f3"])
            e073.V1_60K = Path(cache_paths["e072_v1"])
            e073.V2_60K = Path(cache_paths["e072_v2"])
            e072.E046_CKPT = Path(cache_paths["e046"])
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
