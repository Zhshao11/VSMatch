#!/usr/bin/env python3
"""CPU contract checks for LuckyR1 v2 evidence."""
from __future__ import annotations

import torch

from luckyr1_models_v2 import EvidenceAdapter, SparseGeometryBelief


def coordinates(batch=1, height=16, width=16):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack((xx, yy), dim=0).float()[None]
    return grid.repeat(batch, 1, 1, 1)


def main():
    torch.manual_seed(7)
    corr = torch.randn(1, 324, 16, 16)
    coords0 = coordinates()
    coords1 = coords0 + torch.tensor([1.25, -0.75]).view(1, 2, 1, 1)

    # With both evidence strengths disabled, raw CorrBlock values are unchanged.
    belief = SparseGeometryBelief()
    belief.geometry_strength.data.fill_(-40)
    belief.temporal_strength.data.fill_(-40)
    raw, state, _ = belief(corr, coords0, coords1)
    assert torch.equal(raw, corr)
    assert state["targets"].shape == (1, 16, 16, 4, 2)

    # Absolute target translation must not alter relative geometric evidence.
    belief.geometry_strength.data.fill_(-2)
    out1, state1, _ = belief(corr, coords0, coords1)
    shift = torch.tensor([3.0, -2.0]).view(1, 2, 1, 1)
    out2, state2, _ = belief(corr, coords0, coords1 + shift)
    assert torch.allclose(out1, out2, atol=2e-6, rtol=2e-6)
    expected_shift = shift.permute(0, 2, 3, 1)[..., None, :]
    assert torch.allclose(state1["targets"] + expected_shift, state2["targets"], atol=2e-6)

    # Two recurrent iterations exercise spatial, temporal, and relation gradients.
    adapter = EvidenceAdapter("C")
    source = torch.randn(1, 8, 16, 16)
    target = torch.randn(1, 8, 16, 16)
    first, modes, _ = adapter(corr, source, target, coords0, coords1, None)
    second, _, _ = adapter(corr, source, target, coords0, coords1, modes)
    (first.square().mean() + second.square().mean()).backward()
    gradients = [parameter.grad for parameter in adapter.parameters()]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(float(g.abs().sum()) > 0 for g in gradients)
    assert first.shape == corr.shape and torch.isfinite(first).all()
    print("PASS luckyr1-bc-v2 CPU contracts")


if __name__ == "__main__":
    main()
