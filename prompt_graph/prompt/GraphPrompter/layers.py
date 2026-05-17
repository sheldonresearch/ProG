"""GraphPrompter core propagation layers (KDD 2025).

Ported and simplified from the original reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_scatter import scatter


class BgGraphToSupernodePropagator(nn.Module):
    """Aggregate background-graph nodes into supernode embeddings.

    Uses ``scatter`` reduction (mean / add / max) over edges that connect
    background nodes to their supernode representative.
    """

    def __init__(self, aggr: str = "mean"):
        super().__init__()
        self.aggr = aggr

    def forward(
        self,
        all_node_emb: torch.Tensor,
        supernode_edge_index: torch.Tensor,
        supernode_idx: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # supernode_edge_index: [2, E] where edge[0] = bg node, edge[1] = supernode
        out = scatter(
            src=all_node_emb[supernode_edge_index[0]],
            index=supernode_edge_index[1],
            dim=0,
            reduce=self.aggr,
        )
        return out[supernode_idx]


class SupernodeToBgGraphPropagator(nn.Module):
    """Propagate updated supernode embeddings back to the background graph."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj_sn_attr = nn.Linear(emb_dim, emb_dim)
        self.proj_sn_attr_2 = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        new_supernode_x: torch.Tensor,
        supernode_edge_index: torch.Tensor,
        supernode_idx: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.clone()
        x[supernode_idx] = x[supernode_idx] + self.proj_sn_attr(new_supernode_x)
        x[supernode_edge_index[0]] = x[supernode_edge_index[0]] + self.proj_sn_attr_2(x[supernode_edge_index[1]])
        return x


class MetaGNNLayer(MessagePassing):
    """Single GAT-style layer for bipartite metagraph propagation.

    Uses edge attributes (2-dim for metagraph) and multi-head attention.
    """

    def __init__(
        self,
        edge_attr_dim: int,
        emb_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        aggr: str = "add",
        batch_norm: bool = True,
    ):
        super().__init__(aggr=aggr)
        self.heads = heads
        self.head_dim = emb_dim // heads
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.mlp_kqv = nn.Linear(emb_dim, 3 * emb_dim)
        self.lin_edge = nn.Linear(edge_attr_dim, emb_dim)
        self.att_mlp = nn.Sequential(
            nn.Linear(3 * self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
        )
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim) if batch_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        size=None,
    ) -> torch.Tensor:
        kqv_x = self.mlp_kqv(x)
        out = self.propagate(edge_index, x=kqv_x, edge_attr=edge_attr, size=size)
        out = F.dropout(out, p=self.dropout, training=self.training) + x
        out = self.bn(out)
        return out

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor, ptr, size_i) -> torch.Tensor:
        H, E = self.heads, self.head_dim
        q = x_i[:, : self.emb_dim].reshape(-1, H, E)
        k = x_j[:, self.emb_dim : 2 * self.emb_dim].reshape(-1, H, E) / math.sqrt(E)
        v = x_j[:, 2 * self.emb_dim : 3 * self.emb_dim].reshape(-1, H, E)

        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(edge_attr.shape[0], H, E)

        alpha = self.att_mlp(torch.cat([k, q, F.relu(edge_attr)], dim=-1))
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        attn_output = alpha * v
        attn_output = attn_output.view(attn_output.shape[0], H * E)
        attn_output = self.out_proj(attn_output)
        return attn_output


class MetaGNN(nn.Module):
    """Stack of ``MetaGNNLayer`` for metagraph propagation.

    Supports optional ``back`` (query-to-support) edges and self-loops.
    """

    def __init__(
        self,
        edge_attr_dim: int,
        emb_dim: int,
        heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.0,
        has_final_back: bool = False,
        msg_pos_only: bool = False,
        self_loops: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.num_gnn_layers = n_layers
        self.msg_pos_only = msg_pos_only
        self.self_loops = self_loops
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(
                MetaGNNLayer(
                    emb_dim=emb_dim,
                    heads=heads,
                    edge_attr_dim=edge_attr_dim,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
        self.gnn_layers_back = (
            MetaGNNLayer(
                emb_dim=emb_dim,
                heads=heads,
                edge_attr_dim=edge_attr_dim,
                batch_norm=batch_norm,
            )
            if has_final_back
            else None
        )
        self.gnn_non_linear = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not query_mask.dtype == torch.bool:
            query_mask = query_mask.bool()

        if not self.msg_pos_only:
            support_mask = ~query_mask
        else:
            positives = edge_attr[:, -1] == 1
            support_mask = (~query_mask) & positives

        edge_index_back = edge_index[:, query_mask].flip(0)
        edge_attr_back = edge_attr[query_mask]

        edge_index = torch.cat([edge_index[:, support_mask], edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr[support_mask], edge_attr], dim=0)

        if self.self_loops:
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=torch.tensor([0.0, 0.0]).to(edge_attr.device),
                num_nodes=num_nodes,
            )

        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index, edge_attr=edge_attr)
            if i != self.num_gnn_layers - 1:
                x = self.gnn_non_linear(x)

        if self.gnn_layers_back is not None:
            x = self.gnn_non_linear(x)
            x = self.gnn_layers_back(x, edge_index_back, edge_attr=edge_attr_back)

        return x
