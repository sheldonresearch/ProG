"""ProNoGPrompt: adaptive neighbour-aggregation prompts (KDD 2025).

ProNoG learns hop-specific element-wise prompt weights and a meta-net
(PromptVector) that generates adaptive prompts from aggregated neighbour
embeddings.  Classification is prototype-based (cosine similarity to class
averages).

This port strips the DGL/FAGCN backbone and keeps only the downstream
prompt-tuning components, delegating the initial node embeddings to ProG's
pre-trained GNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptVector(nn.Module):
    """Bottleneck MLP that generates adaptive prompts."""

    def __init__(self, in_size: int, out_size: int, bottleneck_size: int, dropout: float = 0.1, scaling: float = 0.1):
        super().__init__()
        self.down = nn.Linear(in_size, bottleneck_size, bias=True)
        self.up = nn.Linear(bottleneck_size, out_size, bias=True)
        self.scaling = scaling
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(self.act_fn(self.down(self.dropout(x))))
        if self.scaling:
            h = h * self.scaling
        return h


class DownstreamPrompt(nn.Module):
    """Element-wise weight vector (1 × hid_dim)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.weight.data, gain=1.414)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class ProNoGPrompt(nn.Module):
    """ProNoG downstream prompt module.

    Args:
        embeds: Pre-trained node embeddings [N, hid_dim] (detached).
        neighbors: List of 1-hop neighbour index lists per node.
        neighbors_2hop: List of 2-hop neighbour index lists per node.
        nb_classes: Number of output classes.
        hidden_size: Embedding dimension.
        bottleneck_size: Bottleneck dim for PromptVector.
        dropout: Dropout rate.
        use_metanet: Whether to use the PromptVector meta-net.
        multi_prompt: Whether to use separate prompts per hop.
    """

    def __init__(
        self,
        embeds: torch.Tensor,
        neighbors: list,
        neighbors_2hop: list,
        nb_classes: int,
        hidden_size: int = 256,
        bottleneck_size: int = 64,
        dropout: float = 0.05,
        use_metanet: int = 1,
        multi_prompt: int = 1,
    ):
        super().__init__()
        self.embeds = nn.Parameter(embeds, requires_grad=False)
        self.neighbors = neighbors
        self.neighbors_2hop = neighbors_2hop
        self.nb_classes = nb_classes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_metanet = use_metanet
        self.multi_prompt = multi_prompt

        self.selfprompt = DownstreamPrompt(hidden_size)
        if multi_prompt:
            self.neighborsprompt = DownstreamPrompt(hidden_size)
            self.neighbors_2hopprompt = DownstreamPrompt(hidden_size)
        self.metanet = PromptVector(hidden_size, hidden_size, bottleneck_size, dropout)

        self.ave = None  # class prototypes, computed on first train call

    def forward(self, idx: torch.Tensor, train_labels: torch.Tensor | None = None) -> torch.Tensor:
        """Return softmax logits over classes for the nodes in ``idx``.

        If ``train_labels`` is provided, recompute class prototypes.
        """
        device = self.embeds.device
        center = self.embeds[idx]
        center_embeds = self.selfprompt(center)
        rawret = torch.zeros_like(center_embeds)

        for step in range(center.shape[0]):
            nbrs = self.neighbors[idx[step].item()]
            nbrs_2 = self.neighbors_2hop[idx[step].item()]

            tempneighbors = self.embeds[nbrs] if len(nbrs) else torch.zeros((1, self.hidden_size), device=device)
            tempneighbors_2 = self.embeds[nbrs_2] if len(nbrs_2) else torch.zeros((1, self.hidden_size), device=device)

            if self.multi_prompt:
                neighborsembds = self.neighborsprompt(tempneighbors)
                neighbors_2hopembds = self.neighbors_2hopprompt(tempneighbors_2)
            else:
                neighborsembds = self.selfprompt(tempneighbors)
                neighbors_2hopembds = self.selfprompt(tempneighbors_2)

            neighborsembds[neighborsembds != neighborsembds] = 0
            neighbors_2hopembds[neighbors_2hopembds != neighbors_2hopembds] = 0

            neighbor_embbedings = torch.cat((neighborsembds, neighbors_2hopembds), 0)
            sim = torch.cosine_similarity(
                center_embeds[step].unsqueeze(0).unsqueeze(1),
                neighbor_embbedings.unsqueeze(0),
                dim=-1,
            )
            weights = F.softmax(sim, dim=-1)
            weighted_neighbors = torch.mm(weights, neighbor_embbedings)
            center_embeddings = center_embeds[step].unsqueeze(0)

            inputs = torch.add(weighted_neighbors, center_embeddings)
            inputs = F.dropout(inputs, self.dropout, training=self.training)

            if self.use_metanet:
                prompts = self.metanet(inputs)
                rawret[step] = torch.add(prompts, center_embeddings).squeeze(0)
            else:
                rawret[step] = inputs.squeeze(0)

        if train_labels is not None:
            self.ave = _compute_prototypes(rawret, train_labels, self.nb_classes)

        if self.ave is None:
            raise RuntimeError("ProNoGPrompt: class prototypes not computed yet.")

        ret = torch.cosine_similarity(rawret.unsqueeze(1), self.ave.unsqueeze(0), dim=-1)
        return F.softmax(ret, dim=1)


def _compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            prototypes.append(torch.zeros(embeddings.size(1), device=embeddings.device))
        else:
            prototypes.append(embeddings[mask].mean(0))
    return torch.stack(prototypes)
