"""UniPrompt (NeurIPS 2025) light-weight port for ProG.

Core idea: build a *prompt graph* via k-NN on node features, learn
adjustable edge weights for those prompt edges, and **fuse** the prompt
graph with the original graph before feeding it to the (frozen) pre-trained
GNN:

    comb_index, comb_weight = fuse(original_index, original_weight,
                                   prompt_index, prompt_weight, tau)
    embeds = pretrained_gnn(data.x, comb_index, comb_weight)

Because ProG uses standard PyG conv layers, this module is consumed by the
strategy layer rather than inside ``gnn.forward``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops, coalesce, degree


class UniPrompt(nn.Module):
    """Universal graph prompt via k-NN edge adaptation.

    Parameters
    ----------
    x : Tensor [N, feat_dim]
        Initial node features (used to build the k-NN base graph).
    k : int
        Number of neighbours in the prompt graph.
    metric : str
        ``'cosine'`` or ``'euclidean'`` (passed to ``kneighbors_graph``).
    alpha : float
        Sharpness constant for the ELU-based edge-weight re-parameterisation.
    num_nodes : int
        Number of nodes (used for self-loop bookkeeping).
    """

    def __init__(self, x, k, metric="cosine", alpha=6.0, num_nodes=None):
        super().__init__()
        if num_nodes is None:
            num_nodes = x.size(0)
        self.num_nodes = num_nodes
        self.alpha = alpha

        # Build k-NN adjacency on CPU (sklearn does not handle GPU tensors).
        knn_adj = kneighbors_graph(x.detach().cpu().numpy(), k, metric=metric)
        knn_adj = knn_adj.tocoo()
        edge_index = torch.tensor(
            np.vstack([knn_adj.row, knn_adj.col]), dtype=torch.long
        )
        edge_attr = torch.tensor(knn_adj.data, dtype=torch.float32)

        self.base_edge_index = nn.Buffer(edge_index.to(x.device))
        self.edge_weight = nn.Parameter(edge_attr.to(x.device))

    def forward(self):
        """Return the current prompt graph (edge_index, edge_weight)."""
        weights = F.elu(self.edge_weight * self.alpha - self.alpha) + 1
        return self.base_edge_index, weights

    @staticmethod
    def edge_fuse(index_ori, weight_ori, index_pt, weight_pt, tau=0.99):
        """Fuse original and prompt graphs with interpolation factor ``tau``.

        Returns
        -------
        fused_index : LongTensor [2, E]
        fused_weight : Tensor [E]
        """
        weight_ori = weight_ori * tau
        weight_pt = weight_pt * (1.0 - tau)
        comb_index = torch.cat([index_ori, index_pt], dim=1)
        comb_weight = torch.cat([weight_ori.detach(), weight_pt])
        return coalesce(comb_index, comb_weight, reduce="add")


class UniPromptStrategyMixin:
    """Shared helpers for UniPrompt node/graph strategies."""

    @staticmethod
    def _normalize_edge_index(edge_index, num_nodes, device):
        """Add self-loops and symmetrically normalise edge weights."""
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return edge_index, deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight

    @staticmethod
    def _fuse_and_embed(ctx, data, prompt, tau):
        """Return fused (edge_index, edge_weight) and GNN embeddings."""
        device = ctx.device
        num_nodes = data.num_nodes
        orig_index, orig_weight = UniPromptStrategyMixin._normalize_edge_index(
            data.edge_index, num_nodes, device
        )
        pt_index, pt_weight = prompt()
        fused_index, fused_weight = UniPrompt.edge_fuse(
            orig_index, orig_weight, pt_index, pt_weight, tau
        )
        embeds = ctx.gnn(data.x, fused_index, edge_weight=fused_weight)
        return embeds
