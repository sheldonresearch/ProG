"""Smoke tests for data loading."""

from prompt_graph.data import load4graph, load4node


def test_load4node_cora_returns_valid_data():
    data, input_dim, output_dim = load4node("Cora")
    assert input_dim > 0
    assert output_dim > 0
    # Cora has 2708 nodes, 7 classes, 1433 features
    assert input_dim == 1433
    assert output_dim == 7
    assert data.x.shape[0] == 2708


def test_load4graph_mutag_returns_valid_dataset():
    input_dim, output_dim, dataset = load4graph("MUTAG")
    assert input_dim > 0
    assert output_dim > 0
    assert len(dataset) > 0
