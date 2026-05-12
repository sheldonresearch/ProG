"""Backward-compat shim. See ``prompt_graph/model/gnn.py``."""
from .gnn import GNN


class GCN(GNN):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3,
                 JK="last", drop_ratio=0, pool='mean'):
        super().__init__(input_dim, hid_dim, out_dim, num_layer, JK, drop_ratio, pool,
                         conv_type='GCN')
