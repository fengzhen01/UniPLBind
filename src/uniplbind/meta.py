"""Meta-learning network used to estimate dynamic residue weights."""

from __future__ import annotations

import torch
from torch import nn


class VNet(nn.Module):
    """Map a scalar residue loss to a weight in ``(0, 1)``."""

    def __init__(self, hidden_dim: int = 100) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return self.network(losses.reshape(-1, 1)).reshape(-1)

