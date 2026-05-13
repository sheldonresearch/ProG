import os
import tempfile

import torch
from torch_geometric.data import Data

from prompt_graph.data import (
    induced_graph_cache_path,
    induced_graphs,
    load_induced_graphs,
    split_induced_graphs,
)


def _make_test_graph(num_nodes=20, num_features=4):
    """Create a simple connected graph for testing."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
        dtype=torch.long,
    )
    y = torch.randint(0, 3, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


class TestInducedGraphs:
    def test_induced_graphs_basic(self):
        data = _make_test_graph()
        graphs = induced_graphs(data, smallest_size=5, largest_size=10)
        assert len(graphs) == data.num_nodes
        for i, g in enumerate(graphs):
            assert g.x.size(0) >= 5
            assert g.x.size(0) <= 10
            assert g.y == data.y[i].item()

    def test_split_induced_graphs_with_mask(self):
        data = _make_test_graph()
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        train_mask[-5:] = False  # last 5 nodes are test

        with tempfile.TemporaryDirectory() as tmpdir:
            split_induced_graphs(
                data,
                dir_path=tmpdir,
                device="cpu",
                smallest_size=5,
                largest_size=10,
                train_mask=train_mask,
            )
            cache_path = induced_graph_cache_path(
                tmpdir, smallest_size=5, largest_size=10, leak_safe=True
            )
            assert os.path.exists(cache_path)

    def test_load_induced_graphs_round_trip(self):
        data = _make_test_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override induced_graph_dir to use tmpdir
            from prompt_graph.utils import induced_graph_dir

            original = induced_graph_dir
            try:
                # Monkey-patch for this test
                import prompt_graph.utils

                prompt_graph.utils.induced_graph_dir = lambda name: os.path.join(
                    tmpdir, name
                )

                graphs = load_induced_graphs(
                    "TestDataset",
                    data,
                    device="cpu",
                    smallest_size=5,
                    largest_size=10,
                )
                assert len(graphs) == data.num_nodes
                for g in graphs:
                    assert g.x.device == torch.device("cpu")
            finally:
                prompt_graph.utils.induced_graph_dir = original
