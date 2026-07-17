"""The neural decoder model.

Deliberately small: the product constraint is fast CPU inference with an easy
install, not leaderboard accuracy. A per-distance MLP over the flattened
detection-event vector is the simplest architecture that can beat "no decoder"
and gives an honest floor to compare against MWPM. It is the v1 baseline the
benchmark harness is validated with; richer architectures can slot in behind
the same interface.
"""

from __future__ import annotations

import torch
from torch import nn


class SyndromeMLP(nn.Module):
    """MLP mapping detection events to logical-observable flip logits."""

    def __init__(
        self,
        num_detectors: int,
        num_observables: int = 1,
        hidden_sizes: tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout

        layers: list[nn.Module] = []
        width = num_detectors
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, num_observables))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def hyperparameters(self) -> dict:
        return {
            "num_detectors": self.num_detectors,
            "num_observables": self.num_observables,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
        }

    @classmethod
    def from_hyperparameters(cls, hp: dict) -> SyndromeMLP:
        return cls(
            num_detectors=hp["num_detectors"],
            num_observables=hp["num_observables"],
            hidden_sizes=tuple(hp["hidden_sizes"]),
            dropout=hp["dropout"],
        )
