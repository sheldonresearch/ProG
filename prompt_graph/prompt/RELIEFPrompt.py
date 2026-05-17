"""RELIEFPrompt: RL-guided feature perturbation (NeurIPS 2024).

RELIEF uses a PPO agent to select nodes and apply feature perturbations.
This ProG port replaces the full RL loop with a direct learnable feature
perturbation tensor — the core idea of "prompting via feature editing" is
preserved, while the RL policy is distilled into a single differentiable
parameter matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RELIEFPrompt(nn.Module):
    """Learnable feature perturbation for every node.

    Args:
        num_nodes: Number of graph nodes.
        feat_dim: Dimension of node features.
        scale: Initial scale of the perturbation (default 0.1).
    """

    def __init__(self, num_nodes: int, feat_dim: int, scale: float = 0.1):
        super().__init__()
        self.perturbation = nn.Parameter(torch.zeros(num_nodes, feat_dim))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.perturbation)
        self.perturbation.data *= self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learnable perturbation to node features."""
        return x + self.perturbation
