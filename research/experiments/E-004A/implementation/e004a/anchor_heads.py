import torch.nn as nn

from experiments.sparsegeo.e004a.coarse_probe import PRE_FEATURE_DIM


POST_FEATURE_DIM = 15


class _Head(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class PreSearchRecoverabilityHead(_Head):
    def __init__(self):
        super().__init__(PRE_FEATURE_DIM)


class PostSearchVerifier(_Head):
    def __init__(self):
        super().__init__(POST_FEATURE_DIM)
