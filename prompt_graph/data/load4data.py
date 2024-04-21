import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
from ogb.nodeproppred import PygNodePropPredDataset

def node_sample_and_save(data, k, folder, num_classes):
    # 获取标签
    labels = data.y.to('cpu')
    
    # 随机选择90%的数据作为测试集
    num_test = int(0.9 * data.num_nodes)
    test_idx = torch.randperm(data.num_nodes)[:num_test]
    test_labels = labels[test_idx]
    
    # 剩下的10%作为候选训练集
    remaining_idx = torch.randperm(data.num_nodes)[num_test:]
    remaining_labels = labels[remaining_idx]
    
    # 从剩下的数据中选出k*标签数个样本作为训练集
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def graph_sample_and_save(dataset, k, folder, num_classes):

    # 计算测试集的数量（例如90%的图作为测试集）
    num_graphs = len(dataset)
    num_test = int(0.8 * num_graphs)
    labels = torch.tensor([graph.y.item() for graph in dataset])

    # 随机选择测试集的图索引
    all_indices = torch.randperm(num_graphs)
    test_indices = all_indices[:num_test]
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    test_labels = labels[test_indices]
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_indices = all_indices[num_test:]

    # 从剩下的10%的图中为训练集选择每个类别的k个样本
    train_indices = []
    for i in range(num_classes):
        # 选出该类别的所有图
        class_indices = [idx for idx in remaining_indices if labels[idx].item() == i]
        # 如果选出的图少于k个，就取所有该类的图
        selected_indices = class_indices[:k] 
        train_indices.extend(selected_indices)

    # 随机打乱训练集的图索引
    train_indices = torch.tensor(train_indices)
    shuffled_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_indices]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    train_labels = labels[train_indices]
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))

def load4graph(dataset_name, shot_num= 10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        # # 分类并选择每个类别的图
        # class_datasets = {}
        # for data in dataset:
        #     label = data.y.item()
        #     if label not in class_datasets:
        #         class_datasets[label] = []
        #     class_datasets[label].append(data)

        # train_data = []
        # remaining_data = []
        # for label, data_list in class_datasets.items():
        #     train_data.extend(data_list[:shot_num])
        #     random.shuffle(train_data)
        #     remaining_data.extend(data_list[shot_num:])

        # # 将剩余的数据 1：9 划分为测试集和验证集
        # random.shuffle(remaining_data)
        # val_dataset_size = len(remaining_data) // 9
        # val_dataset = remaining_data[:val_dataset_size]
        # test_dataset = remaining_data[val_dataset_size:]
        
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset



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
    
def load4node(dataname):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        graph, labels = dataset[0]
        edge_index = torch.stack(graph.edges())
        x = graph.ndata['feat']
        y = labels.squeeze()
        data = Data(x=x, edge_index = edge_index, y = y)
        input_dim = x.shape[1]
        out_dim = dataset.num_classes

    return data, input_dim, out_dim


    # print()
    # print(f'Dataset: {dataset}:')
    # print('======================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    # data = dataset[0]  # Get the first graph object.

    # print()
    # print(data)
    # print('===========================================================================================================')

    # # Gather some statistics about the graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')

    #  # 根据 shot_num 更新训练掩码
    # class_counts = {}  # 统计每个类别的节点数
    # for label in data.y:
    #     label = label.item()
    #     class_counts[label] = class_counts.get(label, 0) + 1

    
    # train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    
    # for label in data.y.unique():
    #     label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

    #     # if len(label_indices) < 3 * shot_num:
    #     #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

    #     label_indices = label_indices[torch.randperm(len(label_indices))]
    #     train_indices = label_indices[:shot_num]
    #     train_mask[train_indices] = True       
    #     remaining_indices = label_indices[100:]
    #     # split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
        
    #     # val_indices = remaining_indices[:split_point]
    #     test_indices = remaining_indices

    #     # val_mask[val_indices] = True
    #     test_mask[test_indices] = True

    # data.train_mask = train_mask
    # data.test_mask = test_mask
    # # data.val_mask = val_mask


def load4link_prediction_single_graph(dataname, num_per_samples=1):
    data, input_dim, output_dim = load4node(dataname)

    
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
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边
    data = Batch.from_data_list(dataset)
    
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

# used in pre_train.py
def NodePretrain(dataname='CiteSeer', num_parts=200):
    data, input_dim, _ = load4node(dataname)

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    graph_list = list(ClusterData(data=data, num_parts=num_parts))

    return graph_list, input_dim


