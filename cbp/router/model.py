from __future__ import annotations

import torch
from torch import nn


class RouterMLP(nn.Module):
    """Single-image inductive router for a fixed CBP teacher partition."""

    def __init__(
        self,
        input_dim: int,
        n_cells: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_cells = int(n_cells)
        self.hidden = tuple(int(x) for x in hidden)
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        d = self.input_dim
        for h in self.hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)])
            d = h
        layers.append(nn.Linear(d, self.n_cells))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "n_cells": self.n_cells,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
        }
