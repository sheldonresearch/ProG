"""Smoke tests for GNN model factory."""

import pytest
import torch

from prompt_graph.model import GAT, GCN, GIN, GCov, GraphSAGE, GraphTransformer

GNN_CLASSES = [GCN, GAT, GraphSAGE, GIN, GCov, GraphTransformer]


@pytest.mark.parametrize("cls", GNN_CLASSES)
def test_gnn_instantiates(cls):
    """Each GNN backbone should construct without raising."""
    model = cls(input_dim=10, hid_dim=16, num_layer=2)
    assert model is not None
    # Forward smoke
    x = torch.randn(20, 10)
    edge_index = torch.randint(0, 20, (2, 40))
    out = model(x, edge_index)
    assert out is not None
