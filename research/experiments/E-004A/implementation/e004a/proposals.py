from dataclasses import dataclass
from typing import Tuple

import torch

from experiments.sparsegeo.e004a.spec_binding import UNIFORM_REGION_OFFSETS


REGION_COUNT = 64
PROPOSALS_PER_REGION = 4
PROPOSAL_COUNT = REGION_COUNT * PROPOSALS_PER_REGION
SELECTED_COUNT = 128
REGION_OFFSETS: Tuple[Tuple[int, int], ...] = ((2, 2), (2, 6), (6, 2), (6, 6))


@dataclass(frozen=True)
class ProposalSet:
    proposal_id: torch.Tensor
    region_id: torch.Tensor
    region_xy: torch.Tensor
    offset_yx: torch.Tensor
    lattice_xy: torch.Tensor
    lattice_indices: torch.Tensor
    pixel_xy: torch.Tensor

    def index_select(self, indices: torch.Tensor):
        indices = indices.to(torch.long).cpu()
        return ProposalSet(**{
            name: getattr(self, name).index_select(0, indices)
            for name in self.__dataclass_fields__
        })


def build_region_proposals() -> ProposalSet:
    rows = []
    for ry in range(8):
        for rx in range(8):
            region_id = 8 * ry + rx
            for dy, dx in REGION_OFFSETS:
                y = 8 * ry + dy
                x = 8 * rx + dx
                rows.append((len(rows), region_id, rx, ry, dy, dx, x, y))
    values = torch.tensor(rows, dtype=torch.long)
    result = ProposalSet(
        proposal_id=values[:, 0],
        region_id=values[:, 1],
        region_xy=values[:, 2:4],
        offset_yx=values[:, 4:6],
        lattice_xy=values[:, 6:8],
        lattice_indices=values[:, 7] * 64 + values[:, 6],
        pixel_xy=values[:, 6:8] * 8,
    )
    if torch.unique(result.lattice_indices).numel() != PROPOSAL_COUNT:
        raise RuntimeError("E-004A proposal construction produced duplicate tokens")
    return result


def uniform_region_k128(proposals: ProposalSet = None) -> torch.Tensor:
    proposals = build_region_proposals() if proposals is None else proposals
    allowed = torch.tensor(UNIFORM_REGION_OFFSETS, dtype=torch.long)
    mask = (proposals.offset_yx[:, None, :] == allowed[None, :, :]).all(dim=-1).any(dim=-1)
    selected = proposals.proposal_id[mask]
    if selected.numel() != SELECTED_COUNT:
        raise RuntimeError("BIND-001 must select exactly two anchors per region")
    counts = torch.bincount(proposals.region_id[selected], minlength=REGION_COUNT)
    if not bool((counts == 2).all()):
        raise RuntimeError("BIND-001 region allocation is not exactly uniform")
    return selected


def allocate_oracle_k128(
    recoverable: torch.Tensor,
    nearest_error: torch.Tensor,
    proposals: ProposalSet = None,
) -> torch.Tensor:
    proposals = build_region_proposals() if proposals is None else proposals
    recoverable = recoverable.to(torch.bool).reshape(-1).cpu()
    nearest_error = nearest_error.to(torch.float64).reshape(-1).cpu()
    if recoverable.numel() != PROPOSAL_COUNT or nearest_error.numel() != PROPOSAL_COUNT:
        raise ValueError("oracle allocator requires 256 labels/errors")
    if not bool(torch.isfinite(nearest_error).all()):
        raise ValueError("oracle ranking error must be finite")

    def key(index):
        return (not bool(recoverable[index]), float(nearest_error[index]), int(index))

    chosen = []
    counts = [0] * REGION_COUNT
    for region in range(REGION_COUNT):
        members = torch.nonzero(proposals.region_id == region, as_tuple=False).reshape(-1).tolist()
        winner = min(members, key=key)
        chosen.append(winner)
        counts[region] = 1
    remaining = sorted((i for i in range(PROPOSAL_COUNT) if i not in set(chosen)), key=key)
    for index in remaining:
        region = int(proposals.region_id[index])
        if counts[region] >= 3:
            continue
        chosen.append(index)
        counts[region] += 1
        if len(chosen) == SELECTED_COUNT:
            break
    if len(chosen) != SELECTED_COUNT or min(counts) < 1 or max(counts) > 3:
        raise RuntimeError("coverage-constrained oracle allocation failed")
    return torch.tensor(chosen, dtype=torch.long)


def region_counts(selected: torch.Tensor, proposals: ProposalSet = None) -> torch.Tensor:
    proposals = build_region_proposals() if proposals is None else proposals
    selected = selected.to(torch.long).reshape(-1).cpu()
    return torch.bincount(proposals.region_id.index_select(0, selected), minlength=REGION_COUNT)


def allocate_coverage_constrained_k128(scores, proposals: ProposalSet = None):
    proposals = build_region_proposals() if proposals is None else proposals
    scores = scores.to(torch.float64).reshape(-1).cpu()
    if scores.numel() != PROPOSAL_COUNT or not bool(torch.isfinite(scores).all()):
        raise ValueError("allocator requires 256 finite scores")
    order = sorted(range(PROPOSAL_COUNT), key=lambda i: (-float(scores[i]), int(proposals.region_id[i]), i))
    selected, selected_set = [], set()
    counts = [0] * REGION_COUNT
    for region in range(REGION_COUNT):
        winner = next(i for i in order if int(proposals.region_id[i]) == region)
        selected.append(winner); selected_set.add(winner); counts[region] = 1
    for index in order:
        region = int(proposals.region_id[index])
        if index in selected_set or counts[region] >= 3:
            continue
        selected.append(index); selected_set.add(index); counts[region] += 1
        if len(selected) == SELECTED_COUNT:
            break
    if len(selected) != SELECTED_COUNT or min(counts) < 1 or max(counts) > 3:
        raise RuntimeError("coverage allocation failed")
    reserves = [i for i in order if i not in selected_set]
    return torch.tensor(selected, dtype=torch.long), torch.tensor(reserves, dtype=torch.long)
