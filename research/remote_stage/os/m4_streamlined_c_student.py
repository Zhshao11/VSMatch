"""Only the live C/student graph, with the same inference cleanup as E095A.

Construct from the strictly loaded protected M4's c.student on CPU, then
discard the legacy model. No B matcher, teacher or external fusion gate is
registered or executed. The caller applies one terminal official LSR.
"""
import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .m4_clean_deploy import DescriptorAdapters, RecurrentMatcher, parameter_stats, state_digest
else:
    from m4_clean_deploy import DescriptorAdapters, RecurrentMatcher, parameter_stats, state_digest


class M4StreamlinedCStudent(nn.Module):
    method_name = "M4-streamlined-C-student"

    def __init__(self, student):
        super().__init__()
        if student.training or student.variant != "F3":
            raise ValueError("Expected the strictly loaded, eval-mode protected C/student (F3)")
        front = student.b2.e046.front
        if front.grid != 32:
            raise ValueError("Only the protected 32x32 descriptor grid is verified")
        self.dino = front.dino_model
        if self.dino.training or any(p.requires_grad for p in self.dino.parameters()):
            raise ValueError("Frozen eval DINO required")
        self.descriptor_adapters = DescriptorAdapters(front).eval()
        if front.training or any(p.requires_grad for p in self.descriptor_adapters.parameters()):
            raise ValueError("Frozen eval descriptor adapters required")
        self.matcher = RecurrentMatcher(student, 0)
        dx, dy = torch.linspace(-4, 4, 9), torch.linspace(-4, 4, 9)
        self.register_buffer("lookup_offsets", torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1))
        self.audit = {
            "method": self.method_name,
            "original_student": parameter_stats(student),
            "streamlined": parameter_stats(self),
            "frozen_dino_state_sha256": state_digest(self.dino),
            "frozen_descriptor_state_sha256": state_digest(self.descriptor_adapters),
            "counts": {"matchers": 1, "feature_frontends": 2, "correlation_pyramids": 2,
                       "levels_per_pyramid": 4, "dino_forwards": 2, "internal_lsr": 0,
                       "terminal_lsr": 1, "external_fusion_gates": 0},
            "removed": ["B/F2 and B/F3", "B and M4 external fusion gates", "C teachers",
                        "nested unused updates/contexts", "nonfinal masks and upsampling",
                        "unused F3 feature fusion and heads", "unused diagnostics"],
        }
        self.eval()

    @torch.inference_mode()
    def forward(self, source, target, *, iters=32):
        if self.training:
            raise RuntimeError("This is an inference-only model")
        from experiments.sparsegeo.e027a.dino_transfer import materialize_optical_dino_teacher
        resized_cache = {}
        descriptor_pair = None

        def descriptors(group, source_size, target_size):
            nonlocal descriptor_pair
            if group != 0:
                raise ValueError("C/student has exactly one descriptor frontend")
            if descriptor_pair is None:
                dino_pair = tuple(materialize_optical_dino_teacher(self.dino, image, modality,
                                  image.device, 32).detach()
                                  for image, modality in ((source, "optical"), (target, "sar")))
                descriptor_pair = self.descriptor_adapters(target, *dino_pair)
            key = (tuple(source_size), tuple(target_size))
            if key not in resized_cache:
                resized_cache[key] = tuple(F.interpolate(descriptor.float(), size=size,
                                          mode="bilinear", align_corners=False)
                                          for descriptor, size in zip(descriptor_pair,
                                                                     (source_size, target_size)))
            return resized_cache[key]

        return self.matcher(source, target, iters, descriptors, self.lookup_offsets)
