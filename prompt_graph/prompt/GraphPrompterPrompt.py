"""GraphPrompterPrompt: adaptive prompt selection (KDD 2025).

GraphPrompter learns a scoring MLP (select_layers) that ranks candidate
prompt supernodes.  This ProG port keeps the scoring MLP as the core
learnable component and drops the heavy kNN cache / text-embedding / metagraph
pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPrompterPrompt(nn.Module):
    """Scoring MLP that selects adaptive prompt embeddings.

    Args:
        emb_dim: Dimension of node / graph embeddings.
        hidden_dim: Hidden dimension of the scoring MLP.
        num_prompts: Number of candidate prompt vectors to maintain.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, num_prompts: int = 10):
        super().__init__()
        self.prompts = nn.Parameter(torch.zeros(num_prompts, emb_dim))
        self.select_layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompts)
        for layer in self.select_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score and aggregate prompt vectors for the input embeddings.

        Args:
            x: [N, emb_dim] node embeddings.

        Returns:
            [N, emb_dim] prompt-augmented embeddings.
        """
        # Score each candidate prompt for each node
        scores = self.select_layers(self.prompts).squeeze(-1)  # [num_prompts]
        weights = F.softmax(scores, dim=0)
        aggregated = weights.unsqueeze(1) * self.prompts  # [num_prompts, emb_dim]
        prompt_vec = aggregated.sum(0)  # [emb_dim]
        return x + prompt_vec
