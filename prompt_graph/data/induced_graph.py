import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from prompt_graph.utils import get_logger

logger = get_logger(__name__)


def _build_single_induced_graph(
    data,
    center_idx,
    smallest_size,
    largest_size,
    initial_hop=2,
    max_hop=5,
    label=None,
    candidate_mask=None,
):
    """Core logic for building a single node-centered induced subgraph.

    Extracted from ``induced_graphs`` and ``split_induced_graphs`` to avoid
    duplication. Operates on the same device as ``data``.
    """
    if label is None:
        label = int(data.y[center_idx].item())

    current_hop = initial_hop
    subset, _, _, _ = k_hop_subgraph(
        node_idx=center_idx,
        num_hops=current_hop,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    while len(subset) < smallest_size and current_hop < max_hop:
        current_hop += 1
        subset, _, _, _ = k_hop_subgraph(
            node_idx=center_idx,
            num_hops=current_hop,
            edge_index=data.edge_index,
        )

    if len(subset) < smallest_size:
        need_node_num = smallest_size - len(subset)
        label_mask = data.y == label
        if candidate_mask is not None:
            # candidate_mask is intentionally kept on CPU by callers (see
            # split_induced_graphs), but data.y may live on CUDA. Align the
            # masks before AND-ing to avoid a cross-device RuntimeError.
            label_mask = label_mask & candidate_mask.to(label_mask.device)
        pos_nodes = torch.argwhere(label_mask)
        subset_cpu = subset.detach().cpu()
        candidate_nodes = torch.from_numpy(
            np.setdiff1d(pos_nodes.detach().cpu().numpy(), subset_cpu.numpy())
        )
        if candidate_nodes.numel() > 0:
            perm = torch.randperm(candidate_nodes.shape[0])
            take = min(need_node_num, candidate_nodes.shape[0])
            candidate_nodes = candidate_nodes[perm][:take]
            candidate_nodes = candidate_nodes.to(subset.device)
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

    if len(subset) > largest_size:
        subset = subset[torch.randperm(subset.shape[0])][0 : largest_size - 1]
        center_tensor = torch.LongTensor([center_idx]).to(subset.device)
        subset = torch.unique(torch.cat([center_tensor, torch.flatten(subset)]))

    sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
    x = data.x[subset]

    graph = Data(x=x, edge_index=sub_edge_index, y=label)
    return graph


def induced_graphs(data, smallest_size=10, largest_size=30):
    """Basic node-induced subgraph generation (no caching, no device placement)."""
    induced_graph_list = []
    for index in range(data.x.size(0)):
        graph = _build_single_induced_graph(
            data,
            center_idx=index,
            smallest_size=smallest_size,
            largest_size=largest_size,
        )
        induced_graph_list.append(graph)
    return induced_graph_list


def split_induced_graphs(
    data, dir_path, device, smallest_size=10, largest_size=30, train_mask=None
):
    """Production node-induced subgraph generation with disk caching.

    ``train_mask``: optional boolean tensor of length ``data.num_nodes``.
    When provided, padding nodes are drawn **only from training nodes**
    with the same label, preventing test/validation nodes from leaking
    into training subgraphs. Node-level tasks should always provide this.
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
        # WebKB datasets (Wisconsin/Texas/Cornell) ship with multi-split masks
        # of shape (num_nodes, num_splits) — take the first split so downstream
        # AND with label_mask (shape (num_nodes,)) works correctly.
        if train_mask_cpu.dim() > 1:
            train_mask_cpu = train_mask_cpu[:, 0]

    for index in range(data.x.size(0)):
        graph = _build_single_induced_graph(
            data,
            center_idx=index,
            smallest_size=smallest_size,
            largest_size=largest_size,
            candidate_mask=train_mask_cpu,
        )
        graph.index = index
        saved_graph_list.append(deepcopy(graph).to("cpu"))
        induced_graph_list.append(graph)
        if index % 500 == 0:
            logger.info(index)

    os.makedirs(dir_path, exist_ok=True)

    # v2 suffix distinguishes leak-safe caches from legacy leaky ones.
    suffix = "_v2" if train_mask_cpu is not None else ""
    file_path = os.path.join(
        dir_path,
        f"induced_graph_min{smallest_size}_max{largest_size}{suffix}.pkl",
    )
    with open(file_path, "wb") as f:
        pickle.dump(saved_graph_list, f)
        logger.info("induced graph data has been write into " + file_path)


def induced_graph_cache_path(dir_path, smallest_size, largest_size, leak_safe=True):
    """统一的缓存文件名生成，方便调用方与 split_induced_graphs 保持一致。"""
    suffix = "_v2" if leak_safe else ""
    return os.path.join(
        dir_path,
        f"induced_graph_min{smallest_size}_max{largest_size}{suffix}.pkl",
    )


def load_induced_graphs(dataset_name, data, device, smallest_size=100, largest_size=300):
    """Unified induced-graph loader with caching.

    Replaces the copy-pasted ``load_induced_graph`` functions previously
    scattered across ``bench.py``, ``downstream_task.py``, and
    ``node_task.py``.
    """
    from prompt_graph.utils import induced_graph_dir

    folder_path = str(induced_graph_dir(dataset_name))
    os.makedirs(folder_path, exist_ok=True)

    train_mask = getattr(data, "train_mask", None)
    file_path = induced_graph_cache_path(
        folder_path,
        smallest_size=smallest_size,
        largest_size=largest_size,
        leak_safe=train_mask is not None,
    )

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            logger.info("loading induced graph...")
            graphs_list = pickle.load(f)
            logger.info("Done!!!")
    else:
        logger.info("Begin split_induced_graphs.")
        split_induced_graphs(
            data,
            folder_path,
            device,
            smallest_size=smallest_size,
            largest_size=largest_size,
            train_mask=train_mask,
        )
        with open(file_path, "rb") as f:
            graphs_list = pickle.load(f)

    return [graph.to(device) for graph in graphs_list]
