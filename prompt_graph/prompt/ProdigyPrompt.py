"""PRODIGY: Pre-training over Diverse In-Graph Properties (ICML 2023 SPIGM).

A minimal, dependency-light port of the Prodigy metagraph + supernode
propagation idea into the ProG prompt framework.

The original repo (https://github.com/snap-stanford/prodigy) depends on
``transformers`` and ``torch_scatter``.  This module re-implements the
core graph layers with pure PyTorch / PyG so that it runs inside the
standard ProG conda environment.

Architecture (default ``S2,UX,M2`` flavour)
-------------------------------------------
* **S** – Standard GNN (GraphSAGE) on the background graph.
* **U** – ``BgGraphToSupernodePropagator`` (Up) aggregates background-graph
  nodes into a single supernode embedding per subgraph.
* **M** – ``MetaGNN`` (metagraph layer) runs message passing on the bipartite
  graph formed by supernodes and label/query nodes.
* **D** – ``SupernodeToBgGraphPropagator`` (Down) projects the updated
  supernode embeddings back to the background graph.

In the ProG setting ``ProdigyPrompt`` acts as a **graph contextualizer**:
it receives node embeddings from the backbone GNN, builds a lightweight
supernode/metagraph structure, and returns richer contextualized embeddings
that can be fed into the answering head.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

# ---------------------------------------------------------------------------
# Helpers (replace torch_scatter)
# ---------------------------------------------------------------------------

def _scatter_mean(src, index, dim_size):
    """Pure-PyTorch replacement for ``torch_scatter.scatter_mean``."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(0, index.unsqueeze(1), torch.ones_like(index.unsqueeze(1), dtype=src.dtype))
    count = count.clamp(min=1)
    return out / count


def _scatter_sum(src, index, dim_size):
    """Pure-PyTorch replacement for ``torch_scatter.scatter_sum``."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


# ---------------------------------------------------------------------------
# Metagraph layer
# ---------------------------------------------------------------------------

class MetaGNNLayer(MessagePassing):
    """GAT-style message passing for bipartite metagraphs.

    Each node projects itself into key, query, value via a shared linear
    layer.  Edge attributes (2-dim by default) are projected and used in
    the attention MLP.
    """

    def __init__(self, emb_dim, edge_attr_dim=2, heads=8, dropout=0.0, batch_norm=True):
        super().__init__(aggr="add")
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
        self.bn = nn.LayerNorm(emb_dim) if batch_norm else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None, start_right=None):
        kqv_x = self.mlp_kqv(x)
        out = self.propagate(edge_index, x=kqv_x, edge_attr=edge_attr)
        out = F.dropout(out, p=self.dropout, training=self.training) + x
        out = self.bn(out)
        return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        H, E = self.heads, self.head_dim
        q = x_i[:, : self.emb_dim].reshape(-1, H, E)
        k = x_j[:, self.emb_dim : 2 * self.emb_dim].reshape(-1, H, E) / math.sqrt(E)
        v = x_j[:, 2 * self.emb_dim :].reshape(-1, H, E)

        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, H, E)
            alpha = self.att_mlp(torch.cat([k, q, F.relu(edge_attr)], dim=-1))
        else:
            alpha = self.att_mlp(torch.cat([k, q], dim=-1))

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        attn_output = (alpha * v).view(-1, H * E)
        attn_output = self.out_proj(attn_output)
        return attn_output


class MetaGNN(nn.Module):
    """Stack of ``MetaGNNLayer`` s with optional back-propagation edges."""

    def __init__(
        self,
        emb_dim,
        edge_attr_dim=2,
        n_layers=1,
        heads=8,
        dropout=0.0,
        has_final_back=False,
        msg_pos_only=False,
        self_loops=True,
        batch_norm=True,
    ):
        super().__init__()
        self.msg_pos_only = msg_pos_only
        self.self_loops = self_loops
        self.gnn_layers = nn.ModuleList(
            [
                MetaGNNLayer(emb_dim, edge_attr_dim, heads, dropout, batch_norm)
                for _ in range(n_layers)
            ]
        )
        self.gnn_layers_back = (
            MetaGNNLayer(emb_dim, edge_attr_dim, heads, dropout, batch_norm)
            if has_final_back
            else None
        )
        self.gnn_non_linear = nn.GELU()

    def forward(self, x, edge_index, edge_attr, query_mask, start_right):
        if not query_mask.dtype == torch.bool:
            query_mask = query_mask.bool()
        support_mask = (~query_mask) if not self.msg_pos_only else (~query_mask) & (edge_attr[:, -1] == 1)

        query_in_mask = query_mask
        edge_index_back = edge_index[:, query_in_mask].flip(0)
        edge_attr_back = edge_attr[query_in_mask]

        edge_index = torch.cat([edge_index[:, support_mask], edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr[support_mask], edge_attr], dim=0)

        if self.self_loops:
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=torch.tensor([0, 0], device=edge_attr.device),
                num_nodes=num_nodes,
            )

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index, edge_attr=edge_attr, start_right=start_right)
            if i != len(self.gnn_layers) - 1:
                x = self.gnn_non_linear(x)

        if self.gnn_layers_back is not None:
            x = self.gnn_non_linear(x)
            x = self.gnn_layers_back(x, edge_index_back, edge_attr=edge_attr_back, start_right=start_right)

        return x


# ---------------------------------------------------------------------------
# Supernode propagation layers
# ---------------------------------------------------------------------------

class BgGraphToSupernodePropagator(nn.Module):
    """Up layer: aggregate background-graph nodes into supernode embeddings."""

    def __init__(self, aggr="mean"):
        super().__init__()
        self.aggr = aggr

    def forward(self, all_node_emb, supernode_edge_index, supernode_idx, graph_batch=None):
        # supernode_edge_index[0] = bg nodes, [1] = supernode side
        if self.aggr == "mean":
            aggr = _scatter_mean(
                all_node_emb[supernode_edge_index[0]],
                supernode_edge_index[1],
                dim_size=all_node_emb.size(0),
            )
        else:
            aggr = _scatter_sum(
                all_node_emb[supernode_edge_index[0]],
                supernode_edge_index[1],
                dim_size=all_node_emb.size(0),
            )
        return aggr[supernode_idx]


class SupernodeToBgGraphPropagator(nn.Module):
    """Down layer: project supernode embeddings back to the background graph."""

    def __init__(self, emb_dim):
        super().__init__()
        self.proj_sn_attr = nn.Linear(emb_dim, emb_dim)
        self.proj_sn_attr_2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, new_supernode_x, supernode_edge_index, supernode_idx, graph_batch=None):
        x[supernode_idx] = x[supernode_idx] + self.proj_sn_attr(new_supernode_x)
        x[supernode_edge_index[0]] = x[supernode_edge_index[0]] + self.proj_sn_attr_2(x[supernode_edge_index[1]])
        return x


# ---------------------------------------------------------------------------
# Prodigy Prompt (main class)
# ---------------------------------------------------------------------------

class ProdigyPrompt(nn.Module):
    """Prodigy-style in-context graph prompt for ProG.

    This module wraps the metagraph + supernode propagation pipeline into a
    single ``nn.Module`` that can be dropped into a ProG ``NodeTask`` or
    ``GraphTask`` pipeline.

    Parameters
    ----------
    emb_dim : int
        Hidden dimension (must match the backbone GNN output dim).
    layer_spec : str, optional
        Comma-separated layer spec.  Supported tokens:
        ``U`` (Up), ``D`` (Down), ``M`` / ``M2`` (MetaGNN),
        ``UX`` (Up with cat/pool, simplified to Up here).
        Default ``"U,M2,D"`` gives the classic U-M-D sandwich.
    edge_attr_dim : int
        Dimension of edge attributes fed into MetaGNN (default 2).
    meta_heads : int
        Number of attention heads in MetaGNN (default 8).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        emb_dim,
        layer_spec="U,M2,D",
        edge_attr_dim=2,
        meta_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.layer_spec = [s.strip().upper() for s in layer_spec.split(",")]
        self.layers = nn.ModuleList()

        for token in self.layer_spec:
            if token.startswith("M"):
                n = int(token[1:]) if token[1:].isdigit() else 1
                self.layers.append(
                    MetaGNN(
                        emb_dim=emb_dim,
                        edge_attr_dim=edge_attr_dim,
                        n_layers=n,
                        heads=meta_heads,
                        dropout=dropout,
                        has_final_back=False,
                        msg_pos_only=False,
                        self_loops=True,
                        batch_norm=True,
                    )
                )
            elif token == "U" or token == "UX" or token == "UY":
                self.layers.append(BgGraphToSupernodePropagator(aggr="mean"))
            elif token == "D":
                self.layers.append(SupernodeToBgGraphPropagator(emb_dim))
            else:
                raise ValueError(f"Unsupported Prodigy layer token: {token!r}")

        # Learnable prompt tokens appended to the graph during forward.
        # In the original Prodigy these are label / query embeddings;
        # here we keep a small bank of universal prompt vectors.
        self.prompt_token = nn.Parameter(torch.randn(1, emb_dim) * 0.02)

    def forward(self, x, edge_index, batch=None, **kwargs):
        """Run the Prodigy propagation pipeline.

        For ProG compatibility this follows the same signature as other
        prompt modules (``GPF.add``, ``Gprompt`` etc.).

        Parameters
        ----------
        x : Tensor [N, emb_dim]
            Node embeddings from the backbone GNN.
        edge_index : LongTensor [2, E]
            Edge index of the original graph.
        batch : LongTensor [N], optional
            Batch vector (for graph-level tasks).

        Returns
        -------
        Tensor [N, emb_dim]
            Contextualized node embeddings.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Build a trivial supernode per graph in the batch.
        num_graphs = int(batch.max().item()) + 1
        supernode_x = self.prompt_token.repeat(num_graphs, 1)

        # Supernode indices appended at the end of x.
        x = torch.cat([x, supernode_x], dim=0)
        supernode_idx = torch.arange(
            x.size(0) - num_graphs, x.size(0), device=x.device
        )

        # Edges from every node to its graph's supernode.
        supernode_edge_index = torch.stack([torch.arange(x.size(0) - num_graphs, device=x.device), batch], dim=0)

        # Dummy metagraph edges (supernode <-> supernode fully connected).
        if num_graphs > 1:
            rel_idx = torch.arange(num_graphs, device=x.device)
            metagraph_edge_index = torch.combinations(rel_idx, with_replacement=False).t().contiguous()
            # Add reverse edges
            metagraph_edge_index = torch.cat([metagraph_edge_index, metagraph_edge_index.flip(0)], dim=1)
        else:
            metagraph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)

        # Dummy edge attributes [pos/neg indicator, edge_type] for metagraph.
        metagraph_edge_attr = torch.zeros(metagraph_edge_index.size(1), 2, device=x.device)
        metagraph_edge_attr[:, 0] = 1  # positive edges

        # Run layer stack.
        for layer in self.layers:
            if isinstance(layer, BgGraphToSupernodePropagator):
                supernode_x = layer(x, supernode_edge_index, supernode_idx, batch)
            elif isinstance(layer, SupernodeToBgGraphPropagator):
                x = layer(x, supernode_x, supernode_edge_index, supernode_idx, batch)
            elif isinstance(layer, MetaGNN):
                # Metagraph operates only on the supernodes.
                query_mask = torch.zeros(metagraph_edge_index.size(1), dtype=torch.bool, device=x.device)
                supernode_x = layer(
                    supernode_x,
                    metagraph_edge_index,
                    metagraph_edge_attr,
                    query_mask=query_mask,
                    start_right=0,
                )
                # Write updated supernodes back into x.
                x[supernode_idx] = supernode_x
            else:
                raise RuntimeError(f"Unexpected layer type: {type(layer)}")

        # Return only the original nodes (strip supernodes).
        return x[: -num_graphs]
