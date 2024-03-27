import random


def graph_split(graph_list, shot_num):
    r"""A data object describing a homogeneous graph.
      The data object can hold node-level, link-level and graph-level attributes.
      In general, :class:`~torch_geometric.data.Data` tries to mimic the
      behavior of a regular :python:`Python` dictionary.
      In addition, it provides useful functionality for analyzing graph
      structures, and provides basic PyTorch tensor functionalities.
      See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
      introduction.html#data-handling-of-graphs>`__ for the accompanying
      tutorial.

      .. code-block:: python

          from torch_geometric.data import Data

          data = Data(x=x, edge_index=edge_index, ...)

          # Add additional arguments to `data`:
          data.train_idx = torch.tensor([...], dtype=torch.long)
          data.test_mask = torch.tensor([...], dtype=torch.bool)

          # Analyzing the graph structure:
          data.num_nodes
          >>> 23

          data.is_directed()
          >>> False

          # PyTorch tensor functionality:
          data = data.pin_memory()
          data = data.to('cuda:0', non_blocking=True)

      Args:
          x (torch.Tensor, optional): Node feature matrix with shape
              :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
          edge_index (LongTensor, optional): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
          edge_attr (torch.Tensor, optional): Edge feature matrix with shape
              :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
          y (torch.Tensor, optional): Graph-level or node-level ground-truth
              labels with arbitrary shape. (default: :obj:`None`)
          pos (torch.Tensor, optional): Node position matrix with shape
              :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
          time (torch.Tensor, optional): The timestamps for each event with shape
              :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
          **kwargs (optional): Additional attributes.
      """

    class_datasets = {}
    for data in graph_list:
        label = data.y
        if label not in class_datasets:
            class_datasets[label] = []
        class_datasets[label].append(data)

    train_data = []
    remaining_data = []
    for label, data_list in class_datasets.items():
        train_data.extend(data_list[:shot_num])
        random.shuffle(train_data)
        remaining_data.extend(data_list[shot_num:])

    # 将剩余的数据 1：9 划分为测试集和验证集
    random.shuffle(remaining_data)
    val_dataset_size = len(remaining_data) // 9
    val_dataset = remaining_data[:val_dataset_size]
    test_dataset = remaining_data[val_dataset_size:]
    return train_data, test_dataset, val_dataset
