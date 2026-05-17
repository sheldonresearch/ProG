"""DAGPrompT prompt modules (NeurIPS 2024).

Learns layer-wise multi-hop re-weighting vectors and parameterized class
centres.  The backbone must expose ``forward_multihop`` so that intermediate
layer embeddings are available.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DAGPrompt(nn.Module):
    """Multi-hop embedding re-weighter.

    Args:
        dim_list: List of dimensions for each hop's embeddings.
        alpha: Decay factor for the learnable gamma vector.
        use_gamma: Whether to multiply by the learnable gamma vector.
    """

    def __init__(self, dim_list: list[int], alpha: float = 0.5, use_gamma: bool = True):
        super().__init__()
        self.alpha = alpha
        self.hop_range = len(dim_list)
        self.use_gamma = use_gamma
        # All hops are projected to the same target dim so that stacking works
        self.target_dim = dim_list[-1]
        self.projs = nn.ModuleList()
        self.weights = nn.ParameterList()
        for dim in dim_list:
            if dim != self.target_dim:
                self.projs.append(nn.Linear(dim, self.target_dim))
            else:
                self.projs.append(nn.Identity())
            self.weights.append(nn.Parameter(torch.Tensor(1, self.target_dim)))
        if self.hop_range >= 2 and use_gamma:
            gamma = alpha * torch.pow((1 - alpha), torch.arange(self.hop_range, dtype=torch.float))
        else:
            gamma = torch.ones(self.hop_range, dtype=torch.float)
        self.gamma = nn.Parameter(gamma)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        if self.hop_range >= 2 and self.use_gamma:
            gamma = self.alpha * torch.pow((1 - self.alpha), torch.arange(self.hop_range, dtype=torch.float))
            self.gamma.data = gamma
        else:
            self.gamma.data.fill_(1.0)

    def forward(self, node_embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        """``node_embeddings`` is a list of length ``hop_range``."""
        out = []
        for i in range(self.hop_range):
            h = self.projs[i](node_embeddings[i])
            h = h * self.weights[i]
            if self.use_gamma:
                h = h * self.gamma[i]
            out.append(h)
        return out


class ParameterizedMultiHopCenterEmbedding(nn.Module):
    """Learnable residual added to empirical multi-hop class centres."""

    def __init__(self, hop_num: int, label_num: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hop_num, label_num, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, centers: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add learnable residual to each hop's class centres."""
        return [centers[i] + self.weight[i] for i in range(len(centers))]


def center_embedding_multihop(input: torch.Tensor, index: torch.Tensor, label_num: int):
    """Compute empirical class centres for a single hop.

    Args:
        input: [N, hidden_dim] embeddings.
        index: [N] labels.
        label_num: Number of classes.

    Returns:
        centres: [label_num, hidden_dim] mean embedding per class.
        counts: [label_num] number of samples per class.
    """
    device = input.device
    dim = input.size(1)
    centers = torch.zeros(label_num, dim, device=device)
    counts = torch.zeros(label_num, device=device)
    for c in range(label_num):
        mask = index == c
        cnt = mask.sum().item()
        if cnt > 0:
            centers[c] = input[mask].mean(0)
            counts[c] = float(cnt)
    return centers, counts
