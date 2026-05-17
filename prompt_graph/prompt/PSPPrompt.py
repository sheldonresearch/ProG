"""PSPPrompt: prototype-based structural prompt (ICLR 2024).

PSP learns a weight matrix that connects graph nodes to label-prototype
nodes.  In this ProG port the two pre-trained encoders (MLP mask + GCN
context) are replaced by ProG's standard GNN backbone; the only learnable
parameters are the prompt edge-weights and the label prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPPrompt(nn.Module):
    """Simplified PSP prompt for ProG.

    Args:
        num_nodes: Number of graph nodes.
        label_num: Number of classes.
        feature_dim: Dimension of node features / embeddings.
        hidden_dim: Dimension of the prompt hidden state.
    """

    def __init__(self, num_nodes: int, label_num: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.label_num = label_num
        # Learnable edge weights between nodes and label prototypes
        self.weight = nn.Parameter(torch.zeros(num_nodes, label_num))
        # Label prototype embeddings
        self.prototype = nn.Parameter(torch.zeros(label_num, hidden_dim))
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.prototype)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return logits [num_nodes, label_num].

        The prompt augments the original graph with virtual label nodes and
        computes a similarity score between each node and each prototype.
        """
        h = self.feature_proj(x)
        # node-to-prototype similarity weighted by learned structure
        sim = torch.mm(h, self.prototype.t())
        weights = F.softmax(self.weight, dim=1)
        logits = sim * weights
        return logits
