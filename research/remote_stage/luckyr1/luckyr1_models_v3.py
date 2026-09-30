#!/usr/bin/env python3
"""LuckyR1 v3: make injected evidence trainable through historical E074-B."""
from __future__ import annotations

import types

import luckyr1_models_v2 as v2

IMPLEMENTATION_VERSION = "luckyr1-bc-v3-e074b-evidence-graph-20260915"

BCModel = v2.BCModel
F2_CKPT = v2.F2_CKPT
F3_CKPT = v2.F3_CKPT
lsr = v2.lsr
SparseGeometryBelief = v2.SparseGeometryBelief
EvidenceAdapter = v2.EvidenceAdapter
LuckySafeE073 = v2.LuckySafeE073
HistoricalLuckyE073 = v2.HistoricalLuckyE073


class M4Model(v2.M4Model):
    """Clean-lineage M4 using sparse multimode v2 adapters."""

    def __init__(self, *args, lucky_kind=None, **kwargs):
        super().__init__(*args, lucky_kind=lucky_kind, **kwargs)
        if lucky_kind:
            for module in self.modules():
                if isinstance(module, v2.v1.LuckySafeE073):
                    module.evidence = EvidenceAdapter(lucky_kind)


class HistoricalLuckyM4(v2.HistoricalLuckyM4):
    """Historical M4 with gradients enabled only through injected evidence.

    E074-B normally evaluates its frozen F2/F3 teachers under ``no_grad``.
    After evidence adapters are injected into those matchers, that would also
    disconnect their trainable parameters. Base parameters remain frozen, so
    calling the two matchers normally restores only the intended evidence graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def evidence_teacher_preds(branch, source, target, iters):
            return (
                branch.f2(source, target, iters=iters, return_features=False),
                branch.f3(source, target, iters=iters, return_features=False),
            )

        self.model.b.teacher_preds = types.MethodType(evidence_teacher_preds, self.model.b)
