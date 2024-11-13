import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

def node_sample_and_save(data, k, folder, num_classes):
    # 获取标签
    labels = data.y.to('cpu')
    
    # 随机选择90%的数据作为测试集
    num_test = int(0.9 * data.num_nodes)
    if num_test < 1000:
        num_test = int(0.7 * data.num_nodes)
    test_idx = torch.randperm(data.num_nodes)[:num_test]
    test_labels = labels[test_idx]
    
    # 剩下的作为候选训练集
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

    # 计算测试集的数量（例如80%的图作为测试集）
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

def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        # 计算所有节点的度数，这将返回一个张量
        deg = degree(data.edge_index[0], dtype=torch.long)

        # 将度数张量变形为[nodes, 1]以便与其他特征拼接
        deg = deg.view(-1, 1).float()
        
        # 如果原始数据没有节点特征，可以直接使用度数作为特征
        if data.x is None:
            data.x = deg
        else:
            # 将度数特征拼接到现有的节点特征上
            data.x = torch.cat([data.x, deg], dim=1)

def load4graph(dataset_name, shot_num= 10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)  # use_node_attr=False时，节点属性为one-hot编码的节点类别
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)        

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
        

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset  # 统一下游任务返回参数的顺序
        
    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        graph_list = [g for g in graph_list]
        node_degree_as_features(graph_list)
        input_dim = graph_list[0].x.size(1)

        for g in graph_list:
            g.y = g.y.squeeze(0)

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return  input_dim, out_dim, dataset
    else:
        raise ValueError(f"Unsupported GraphTask on dataset: {dataset_name}.")
    
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
    elif dataname in ['Wisconsin', 'Texas']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    elif dataname in ['ENZYMES', 'PROTEINS']:
        # 实现TUDataset中两个multi graphs dataset的节点分类
        dataset = TUDataset(root='data/TUDataset', name=dataname, use_node_attr=True)
        node_class = dataset.data.x[:,-3:]
        input_dim = dataset.num_node_features
        out_dim = dataset.num_node_labels
        data = Batch.from_data_list(dataset)  # 将dataset中小图合并成一个大图
        data.y = node_class.nonzero().T[1]
    else:
        raise ValueError(f"Unsupported NodeTask on dataset: {dataname}.")
    return data, input_dim, out_dim


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
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)

    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)
        for g in dataset:
            g.y = g.y.squeeze(1)

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

# 未完待续，需要重写一个能够对large-scale图分类数据集的划分代码，避免node-level和edge-level的预训练算法或prompt方法显存溢出的问题
def load4link_prediction_multi_large_scale_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    dataset = [g for g in dataset]
    node_degree_as_features(dataset)
    input_dim = dataset[0].x.size(1)
    for g in dataset:
        g.y = g.y.squeeze(1)

    batch_graph_num = 20000
    split_num = int(len(dataset)/batch_graph_num)
    data_list = []
    edge_label_list = []
    edge_index_list = []
    for i in range(split_num+1):
        if(i==0):
            data = Batch.from_data_list(dataset[0:batch_graph_num])
        elif(i<=split_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        elif len(dataset)>((i-1)*batch_graph_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        

        data_list.append(data)
        
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
def NodePretrain(data, num_parts=200, split_method='Random Walk'):

    # if(dataname=='Cora'):
    #     num_parts=220
    # elif(dataname=='Texas'):
    #     num_parts=20
    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

    else:
        print('None split method!')
        exit()
    
    return graph_list


