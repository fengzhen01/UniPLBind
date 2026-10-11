"""Core UniPLBind prediction network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSelfAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.scale = output_dim**-0.5

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        query = self.query(features)
        key = self.key(features)
        value = self.value(features)
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        positions = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        valid_keys = positions < lengths.to(features.device).unsqueeze(1)
        scores = scores.masked_fill(~valid_keys.unsqueeze(1), float("-inf"))
        attention = F.softmax(scores.float(), dim=-1).to(value.dtype)
        return torch.bmm(attention, value) + value


class MultiScaleFeatureExtractor(nn.Module):
    """Combine 1-, 3-, and 5-residue convolutional contexts with attention."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(input_dim, 512, kernel, padding="same"),
                        nn.Conv1d(512, 256, kernel, padding="same"),
                        nn.Conv1d(256, 128, kernel, padding="same"),
                    ]
                )
                for kernel in (1, 3, 5)
            ]
        )
        self.attention = nn.ModuleList(
            [MaskedSelfAttention(512), MaskedSelfAttention(256), MaskedSelfAttention(128)]
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, residue_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        branch_values = [residue_features.transpose(1, 2) for _ in self.branches]
        skip_sum = 0.0
        for layer_index in range(3):
            branch_values = [
                self.activation(branch[layer_index](value))
                for branch, value in zip(self.branches, branch_values)
            ]
            merged = sum(branch_values).transpose(1, 2)
            skip_sum = skip_sum + self.attention[layer_index](merged, lengths)
        return merged + skip_sum


class UniPLBind(nn.Module):
    """Residue-level two-class predictor for concatenated ProstT5 and Ankh features."""

    def __init__(self, input_dim: int = 2560, dropout: float = 0.5) -> None:
        super().__init__()
        self.feature_extractor = MultiScaleFeatureExtractor(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, residue_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(residue_features, lengths))

