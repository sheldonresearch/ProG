"""SelfProPrompt: projector MLP for Self-Pro (ICML 2024).

Self-Pro does not use a traditional learnable prompt token.  Instead it
fine-tunes a small projector (MLP) on top of frozen GNN embeddings computed
from an *identity graph* (self-loop only).  This module implements that
projector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfProPrompt(nn.Module):
    """MLP projector used by Self-Pro during downstream prompt-tuning.

    Args:
        input_dim: Dimension of GNN node embeddings.
        output_dim: Dimension of projected embeddings (usually same as input).
        num_layers: Number of linear layers in the projector.
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.linears[i](h))
        h = self.linears[self.num_layers - 1](h)
        return h
