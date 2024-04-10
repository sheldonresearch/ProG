import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import os

def node_sample_and_save(data, k, folder):
    # 获取标签
    labels = data.y
    
    # 随机选择90%的数据作为测试集
    num_test = int(0.9 * data.num_nodes)
    test_idx = torch.randperm(data.num_nodes)[:num_test]
    test_labels = labels[test_idx]
    
    # 剩下的10%作为候选训练集
    remaining_idx = torch.randperm(data.num_nodes)[num_test:]
    remaining_labels = labels[remaining_idx]
    
    # 从剩下的数据中选出k*标签数个样本作为训练集
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(data.num_classes)])
    train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def load4graph(dataset_name, shot_num= 10, num_parts=None):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        # 分类并选择每个类别的图
        class_datasets = {}
        for data in dataset:
            label = data.y.item()
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
        
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        return input_dim, out_dim, train_data, test_dataset, val_dataset, graph_list



    if  dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts=200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
        dataset = list(ClusterData(data=data, num_parts=num_parts))
        graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, None, None, None, graph_list
    
def load4node(dataname, shot_num= 10):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

     # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    
    for label in data.y.unique():
        label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

        # if len(label_indices) < 3 * shot_num:
        #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

        label_indices = label_indices[torch.randperm(len(label_indices))]
        train_indices = label_indices[:shot_num]
        train_mask[train_indices] = True       
        remaining_indices = label_indices[100:]
        # split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
        
        # val_indices = remaining_indices[:split_point]
        test_indices = remaining_indices

        # val_mask[val_indices] = True
        test_mask[test_indices] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    # data.val_mask = val_mask

    return data,dataset

def load4link_prediction_single_graph(dataname, num_per_samples=1):
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim

def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    graph_list = []

    for data in dataset:
        # 对每个图进行处理
        if data.is_directed():
            row, col = data.edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = data.edge_index

        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.num_edges * num_per_samples,
        )

        edge_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

        graph_list.append((data, edge_label, edge_index))

    return graph_list, input_dim, output_dim

# used in pre_train.py
def NodePretrain(dataname='CiteSeer', num_parts=200):
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    data = dataset[0]

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    graph_list = list(ClusterData(data=data, num_parts=num_parts))

    return graph_list, input_dim


