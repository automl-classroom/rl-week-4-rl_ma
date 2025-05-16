from typing import Sequence

from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Ein flexibles MLP, das beliebig viele Hidden-Layer unterstützt.
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (64, 64)
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_sizes : list/tuple of int
            Größen der Hidden-Layer.
        """
        super().__init__()

        layers = []
        in_dim = obs_dim
        # Hidden-Layer dynamisch hinzufügen
        for i, h in enumerate(hidden_sizes):
            layers.append((f"fc{i + 1}", nn.Linear(in_dim, h)))
            layers.append((f"relu{i + 1}", nn.ReLU()))
            in_dim = h
        # Output-Layer
        layers.append(("out", nn.Linear(in_dim, n_actions)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
