import torch
import torch.nn.functional as F

from experiments.sparsegeo.e002a.matcher import flatten_features
from experiments.sparsegeo.e004a.proposals import build_region_proposals


PRE_FEATURE_DIM = 11


def build_coarse_target(target_feature):
    if tuple(target_feature.shape) != (1, 256, 64, 64):
        raise ValueError("target feature must be [1,256,64,64]")
    return F.normalize(F.avg_pool2d(target_feature, kernel_size=4, stride=4), p=2, dim=1)


def _distinctiveness(source_feature, lattice_xy):
    tokens = F.normalize(flatten_features(source_feature)[0], p=2, dim=-1)
    xy = lattice_xy.to(source_feature.device)
    center_indices = xy[:, 1] * 64 + xy[:, 0]
    offsets = torch.tensor((-65, -64, -63, -1, 1, 63, 64, 65), device=source_feature.device)
    neighbor_indices = center_indices[:, None] + offsets[None, :]
    centers = tokens.index_select(0, center_indices)
    neighbors = tokens.index_select(0, neighbor_indices.reshape(-1)).reshape(256, 8, 256)
    values = torch.einsum("nc,nkc->nk", centers, neighbors)
    return torch.stack((1.0 - values.max(dim=-1).values, 1.0 - values.mean(dim=-1)), dim=-1)


def compute_coarse_probe_features(source_feature, target_feature, temperature=0.1):
    proposals = build_region_proposals()
    source_tokens = flatten_features(source_feature)[0]
    source_rows = source_tokens.index_select(0, proposals.lattice_indices.to(source_feature.device))
    source_norm = torch.linalg.vector_norm(source_rows, dim=-1).clamp_min(1e-12)
    coarse = build_coarse_target(target_feature)
    coarse_tokens = flatten_features(coarse)[0]
    scores = F.normalize(source_rows, p=2, dim=-1) @ coarse_tokens.transpose(0, 1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    top = torch.gather(scores, 1, order[:, :4])
    top1_index = order[:, 0]
    probability = torch.softmax(scores / temperature, dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    top4_mass = torch.gather(probability, 1, order[:, :4]).sum(dim=-1)
    coarse_x = top1_index.remainder(16).to(torch.float32)
    coarse_y = torch.div(top1_index, 16, rounding_mode="floor").to(torch.float32)
    coarse_xy = torch.stack((4.0 * coarse_x + 1.5, 4.0 * coarse_y + 1.5), dim=-1)
    xy_norm = 2.0 * proposals.lattice_xy.to(source_feature.device, torch.float32) / 63.0 - 1.0
    features = torch.cat((
        source_norm.log().unsqueeze(-1),
        _distinctiveness(source_feature, proposals.lattice_xy).to(source_feature.device),
        xy_norm,
        top[:, :1],
        (top[:, 0] - top[:, 1]).unsqueeze(-1),
        entropy.unsqueeze(-1),
        top4_mass.unsqueeze(-1),
        coarse_xy / 63.0,
    ), dim=-1).to(torch.float32)
    if features.shape != (256, PRE_FEATURE_DIM):
        raise RuntimeError(f"unexpected pre feature shape {features.shape}")
    return features, coarse_xy.to(torch.float32), scores
