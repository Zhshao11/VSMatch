"""Inference-only M4 with only live modules and per-image frozen-DINO reuse.

Construct from a strictly loaded protected model. Learned active weights and
all recurrent coordinate updates are preserved; this is not a training model.
"""
import hashlib

import torch
from torch import nn
import torch.nn.functional as F


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_stats(module):
    return {"parameters": sum(p.numel() for p in module.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters()),
            "buffers_bytes": sum(p.numel() * p.element_size() for p in module.buffers())}


class FeatureFrontend(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        feature = legacy.feature
        if feature.cfg.feat_cross_attn:
            raise ValueError("Unverified legacy feat_cross_attn path")
        self.opt_adapt, self.sar_adapt = legacy.opt_adapt, legacy.sar_adapt
        self.feat_encoder = feature.feat_encoder
        self.channel_convertor = feature.channel_convertor
        self.transformer = feature.transformer
        self.attn_splits, self.feature_channels = feature.attn_splits, feature.feature_channels

    def forward(self, source, target):
        from core.GDROSnet.FeatureFormer.utils import feature_add_position
        source = self.opt_adapt(source).contiguous()
        target = self.sar_adapt(target).contiguous()
        features = self.channel_convertor(self.feat_encoder(torch.cat([source, target], dim=0)))
        n = features.shape[0] // 2
        source, target = feature_add_position(features[:n], features[n:],
                                              self.attn_splits, self.feature_channels)
        return self.transformer(source, target, attn_num_splits=self.attn_splits)


class DescriptorAdapters(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        if type(legacy).__name__ not in {"DriftControlledDualAdapter", "DualAdapterDinoTransfer"}:
            raise ValueError(f"Unaudited frontend: {type(legacy).__name__}")
        if legacy.grid != 32:
            raise ValueError("Only the protected 32x32 DINO descriptor grid is verified")
        self.optical_adapter, self.sar_adapter = legacy.optical_adapter, legacy.sar_adapter
        self.variant = getattr(legacy, "frontend_variant", "teacher_consistency")
        if self.variant == "common_unique_split":
            self.common_head = legacy.common_head

    def forward(self, sar, optical_dino, sar_dino):
        optical = self.optical_adapter(optical_dino)
        target = self.sar_adapter(sar, sar_dino)
        if self.variant == "common_unique_split":
            optical = F.normalize(optical + 0.05 * torch.tanh(self.common_head(optical)), p=2, dim=1)
            target = F.normalize(target + 0.05 * torch.tanh(self.common_head(target)), p=2, dim=1)
        return optical, target


class RecurrentMatcher(nn.Module):
    def __init__(self, legacy, frontend_group):
        super().__init__()
        if legacy.variant not in {"F2", "F3"}:
            raise ValueError(legacy.variant)
        self.variant, self.frontend_group = legacy.variant, frontend_group
        self.v1 = FeatureFrontend(legacy.b1.front_v1)
        self.v2 = FeatureFrontend(legacy.b2.front_v1)
        self.film_projection = legacy.b2.e046.proj
        self.context = legacy.context
        self.motion_encoder = legacy.update.encoder
        self.gru = legacy.update.gru
        self.flow_head = legacy.update.flow_head
        self.mask_head = legacy.update.mask
        if self.variant == "F2":
            self.fuse_proj, self.fuse_gate = legacy.fuse_proj, legacy.fuse_gate
        else:
            self.corr_gate_logit = legacy.corr_gate_logit

    @staticmethod
    def upsample(flow, mask):
        n, _, h, w = flow.shape
        mask = torch.softmax(mask.view(n, 1, 9, 8, 8, h, w), dim=2)
        up = F.unfold(8 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        return torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3).reshape(n, 2, 8*h, 8*w)

    def forward(self, raw1, raw2, iters, descriptor_pair, offsets):
        from core.corr import CorrBlock
        from core.utils.utils import bilinear_sampler, coords_grid
        image1, image2 = 2 * (raw1 / 255.0) - 1.0, 2 * (raw2 / 255.0) - 1.0
        f1s, f1t = self.v1(image1, image2)
        f2s, f2t = self.v2(image1, image2)
        ds, dt = descriptor_pair(self.frontend_group, f2s.shape[-2:], f2t.shape[-2:])
        gs, bs = self.film_projection(ds).chunk(2, 1)
        gt, bt = self.film_projection(dt).chunk(2, 1)
        f2s = f2s + 0.05 * (torch.tanh(gs) * f2s + bs)
        f2t = f2t + 0.05 * (torch.tanh(gt) * f2t + bt)
        if self.variant == "F2":
            gs = torch.sigmoid(self.fuse_gate(torch.cat([f1s, f2s], dim=1)))
            gt = torch.sigmoid(self.fuse_gate(torch.cat([f1t, f2t], dim=1)))
            fs = (1 - gs) * f1s + gs * self.fuse_proj(torch.cat([f1s, f2s], dim=1))
            ft = (1 - gt) * f1t + gt * self.fuse_proj(torch.cat([f1t, f2t], dim=1))
            pyramids = [CorrBlock(fs.float(), ft.float(), radius=4)]
        else:
            pyramids = [CorrBlock(f1s.float(), f1t.float(), radius=4),
                        CorrBlock(f2s.float(), f2t.float(), radius=4)]
            alpha = torch.sigmoid(self.corr_gate_logit)

        def lookup(pyramid, coords):
            coords = coords.permute(0, 2, 3, 1)
            n, h, w, _ = coords.shape
            responses = []
            for level, corr in enumerate(pyramid.corr_pyramid):
                center = coords.reshape(n*h*w, 1, 1, 2) / 2**level
                sampled = bilinear_sampler(corr, center + offsets.view(1, 9, 9, 2))
                responses.append(sampled.view(n, h, w, -1))
            return torch.cat(responses, dim=-1).permute(0, 3, 1, 2).contiguous().float()

        net, context = torch.split(self.context(image1), [128, 128], dim=1)
        net, context = torch.tanh(net), torch.relu(context)
        n, _, h, w = image1.shape
        c0 = coords_grid(n, h//8, w//8, device=image1.device)
        c1 = coords_grid(n, h//8, w//8, device=image1.device)
        for _ in range(iters):
            c1 = c1.detach()
            corr = lookup(pyramids[0], c1)
            if self.variant == "F3":
                corr = (1 - alpha) * corr + alpha * lookup(pyramids[1], c1)
            flow = c1 - c0
            motion = self.motion_encoder(flow, corr)
            net = self.gru(net, torch.cat([context, motion], dim=1))
            c1 = c1 + self.flow_head(net)
        mask = .25 * self.mask_head(net)
        return self.upsample(c1 - c0, mask)


class CleanM4(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        if legacy.training or legacy.b.variant != "B" or legacy.mode != "region":
            raise ValueError("Expected the protected B/C region M4 in eval mode")
        matchers = [legacy.b.f2, legacy.b.f3, legacy.c.student]
        dinos, groups, memberships = [], {}, []
        adapters, adapter_groups, adapter_dino_groups = [], {}, []
        for name, matcher in zip(("b.f2", "b.f3", "c.student"), matchers):
            front = matcher.b2.e046.front
            dino = front.dino_model
            if any(p.requires_grad for p in dino.parameters()) or dino.training:
                raise ValueError("DINO reuse requires frozen eval backbones")
            digest = state_digest(dino)
            key = (type(dino).__qualname__, repr(dino), digest, front.grid)
            if key not in groups:
                groups[key] = len(dinos)
                dinos.append(dino)
            adapter = DescriptorAdapters(front).eval()
            if any(p.requires_grad for p in adapter.parameters()) or front.training:
                raise ValueError("Descriptor reuse requires frozen eval adapters")
            adapter_digest = state_digest(adapter)
            adapter_key = (groups[key], repr(adapter), adapter.variant, adapter_digest)
            if adapter_key not in adapter_groups:
                adapter_groups[adapter_key] = len(adapters)
                adapters.append(adapter)
                adapter_dino_groups.append(groups[key])
            memberships.append({"matcher": name, "group": groups[key], "state_sha256": digest,
                                "adapter_group": adapter_groups[adapter_key],
                                "adapter_state_sha256": adapter_digest,
                                "frontend_variant": getattr(front, "frontend_variant", None)})
        self.dinos = nn.ModuleList(dinos)
        self.descriptor_adapters = nn.ModuleList(adapters)
        self.adapter_dino_groups = adapter_dino_groups
        self.f2, self.f3, self.student = [RecurrentMatcher(matcher, item["adapter_group"])
                                        for matcher, item in zip(matchers, memberships)]
        self.b_gate, self.m_gate = legacy.b.gate, legacy.gate
        dx, dy = torch.linspace(-4, 4, 9), torch.linspace(-4, 4, 9)
        self.register_buffer("lookup_offsets", torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1))
        self.audit = {"original": parameter_stats(legacy), "clean": parameter_stats(self),
                      "dino_groups": memberships, "unique_dino_backbones": len(dinos),
                      "unique_descriptor_frontends": len(adapters),
                      "removed": ["C teacher module storage", "nested V1/V2 GRUs and ignored contexts",
                                  "unused cost perceiver encoders", "training-only frontend heads",
                                  "F3 feature-fusion modules and unused scalars", "nonfinal mask heads/upsampling",
                                  "discarded F3 feature averages and diagnostic CPU transfers",
                                  "repeated correlation lookup offset construction/transfer"],
                      "reused": ["identical frozen DINO per modality and image pair",
                                 "identical frozen optical/SAR descriptor adapters and interpolation"],
                      "preserved": ["three 32-step recurrent matchers", "five correlation pyramids",
                                    "all active learned adapters/features/gates", "five official LSR fits"]}
        self.eval()

    @torch.inference_mode()
    def forward(self, source, target, *, iters=32, lsr_fn):
        if self.training:
            raise RuntimeError("CleanM4 is an inference-only module")
        from experiments.sparsegeo.e027a.dino_transfer import materialize_optical_dino_teacher
        dino_cache, adapter_cache, resized_cache = {}, {}, {}

        def descriptors(group, source_size, target_size):
            dino_group = self.adapter_dino_groups[group]
            if dino_group not in dino_cache:
                dino_cache[dino_group] = tuple(materialize_optical_dino_teacher(self.dinos[dino_group], image, modality,
                                      image.device, 32).detach()
                                     for image, modality in ((source, "optical"), (target, "sar")))
            if group not in adapter_cache:
                adapter_cache[group] = self.descriptor_adapters[group](target, *dino_cache[dino_group])
            key = (group, tuple(source_size), tuple(target_size))
            if key not in resized_cache:
                resized_cache[key] = tuple(F.interpolate(descriptor.float(), size=size,
                                          mode="bilinear", align_corners=False)
                                          for descriptor, size in zip(adapter_cache[group],
                                                                       (source_size, target_size)))
            return resized_cache[key]

        f2 = self.f2(source, target, iters, descriptors, self.lookup_offsets)
        f3 = self.f3(source, target, iters, descriptors, self.lookup_offsets)
        d = f2 - f3
        l2, l3 = lsr_fn(f2).detach().to(f2.device), lsr_fn(f3).detach().to(f3.device)
        features = torch.cat([f2.detach(), f3.detach(), d.detach(),
                              torch.linalg.vector_norm(d.detach(), dim=1, keepdim=True),
                              torch.linalg.vector_norm((f2-l2).detach(), dim=1, keepdim=True),
                              torch.linalg.vector_norm((f3-l3).detach(), dim=1, keepdim=True)], dim=1)
        g = self.b_gate(features)
        b = (1 - g) * f3 + g * f2
        c = self.student(source, target, iters, descriptors, self.lookup_offsets)
        lb, lc = lsr_fn(b).detach().to(b.device), lsr_fn(c).detach().to(c.device)
        features = torch.cat([b.detach(), c.detach(), (b-c).detach(), torch.abs(b-c).detach(),
                              lb, lc, torch.abs(lb-b).detach(), torch.abs(lc-c).detach()], dim=1)
        g = self.m_gate(features)
        return (1 - g) * b + g * c
