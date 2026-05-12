# from collections import defaultdict
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from prompt_graph.utils import get_logger

logger = get_logger(__name__)


def induced_graphs(data, smallest_size=10, largest_size=30):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(
            node_idx=index, num_hops=current_hop, edge_index=data.edge_index, relabel_nodes=True
        )

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(
                node_idx=index, num_hops=current_hop, edge_index=data.edge_index
            )

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label))
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][
                0:need_node_num
            ]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0 : largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)
        # print(index)
    return induced_graph_list


def split_induced_graphs(
    data, dir_path, device, smallest_size=10, largest_size=30, train_mask=None
):
    """
    将一张大图按节点切成 induced subgraph，并把结果落盘到 dir_path。

    train_mask: 可选的布尔张量，长度等于 data.num_nodes。如果给出，邻域不足时
    只从 train_mask=True 的节点里补齐，避免把测试/验证节点采进训练子图、
    把标签信息泄漏到训练里。**节点任务下必须提供。**
    """

    induced_graph_list = []
    saved_graph_list = []
    from copy import deepcopy

    if train_mask is None:
        import warnings

        warnings.warn(
            "split_induced_graphs() called without train_mask: 邻域补齐会从所有节点"
            "（含 val/test）中采样，存在标签泄漏风险。建议传入 data.train_mask。",
            stacklevel=2,
        )
        train_mask_cpu = None
    else:
        train_mask_cpu = train_mask.detach().to("cpu").bool()
    labels_cpu = data.y.detach().to("cpu")

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(
            node_idx=index, num_hops=current_hop, edge_index=data.edge_index, relabel_nodes=True
        )
        subset = subset

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(
                node_idx=index, num_hops=current_hop, edge_index=data.edge_index
            )

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            label_mask = labels_cpu == int(current_label)
            if train_mask_cpu is not None:
                label_mask = label_mask & train_mask_cpu
            pos_nodes = torch.argwhere(label_mask)
            subset = subset.to("cpu")
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][
                0:need_node_num
            ]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0 : largest_size - 1]
            subset = torch.unique(
                torch.cat([torch.LongTensor([index]).to(device), torch.flatten(subset)])
            )

        subset = subset.to(device)
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
        sub_edge_index = sub_edge_index.to(device)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index=index)
        saved_graph_list.append(deepcopy(induced_graph).to("cpu"))
        induced_graph_list.append(induced_graph)
        if index % 500 == 0:
            logger.info(index)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 给文件名加 _v2 后缀：v1 是旧的、有泄漏的缓存；提供 train_mask 走 v2，
    # 避免把旧 pickle 误当成新算法的结果加载回来。
    suffix = "_v2" if train_mask_cpu is not None else ""
    file_path = os.path.join(
        dir_path,
        f"induced_graph_min{smallest_size}_max{largest_size}{suffix}.pkl",
    )
    with open(file_path, "wb") as f:
        # Assuming 'data' is what you want to pickle
        # pickle.dump(induced_graph_list, f)
        pickle.dump(saved_graph_list, f)
        logger.info("induced graph data has been write into " + file_path)


def induced_graph_cache_path(dir_path, smallest_size, largest_size, leak_safe=True):
    """统一的缓存文件名生成，方便调用方与 split_induced_graphs 保持一致。"""
    suffix = "_v2" if leak_safe else ""
    return os.path.join(
        dir_path,
        f"induced_graph_min{smallest_size}_max{largest_size}{suffix}.pkl",
    )
