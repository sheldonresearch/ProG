"""Unified GNN backbone parameterized by ``conv_type``.

Phase 3 refactor (see ``Docs/IMPROVEMENTS.md`` §2.1/§2.2): the six per-conv
files under ``prompt_graph/model/`` were byte-equivalent except for the conv
class and a single ``GraphConv = <ConvClass>`` line.  They now collapse into
this single module + a registry, and the six original modules are kept as
thin backward-compatible shims.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn import GraphConv as PygGraphConv

from prompt_graph.utils import act


def _make_gin_conv(in_dim: int, out_dim: int) -> nn.Module:
    """Closure used as the conv factory for GIN (matches the legacy lambda)."""
    return GINConv(nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    ))


_GNN_REGISTRY = {
    'GCN': GCNConv,
    'GAT': GATConv,
    'GraphSAGE': SAGEConv,
    'GIN': _make_gin_conv,
    'GCov': PygGraphConv,
    'GraphTransformer': TransformerConv,
}


class GNN(torch.nn.Module):
    """Unified GNN backbone parameterized by ``conv_type``.

    Replaces 6 nearly-identical files (GCN, GAT, GraphSAGE, GIN, GCov,
    GraphTransformer).  Logic is copied verbatim from the legacy
    ``prompt_graph/model/GCN.py`` so behavior is byte-equivalent.
    """

    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3,
                 JK="last", drop_ratio=0, pool='mean', conv_type='GCN'):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            conv_type (str): one of GCN, GAT, GraphSAGE, GIN, GCov, GraphTransformer.

        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """
        if conv_type not in _GNN_REGISTRY:
            raise ValueError(
                f'Unknown conv_type {conv_type!r}; expected one of {sorted(_GNN_REGISTRY)}'
            )
        GraphConv = _GNN_REGISTRY[conv_type]
        self.conv_type = conv_type

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        if batch == None:
            return node_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def build_gnn(name: str, input_dim: int, hid_dim: int, num_layer: int = 2) -> GNN:
    """Factory used by ``BaseTask`` / ``PreTrain`` to replace the six-way dispatch."""
    return GNN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer, conv_type=name)
