import numpy as np
import random
import torch
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch import nn, optim
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from sklearn.metrics import accuracy_score




def evaluate(model, data, nid, batch_size, device,sample_list):
    valid_loader = NeighborSampler(data.edge_index, node_idx=nid, sizes=sample_list, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=0)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad(): 
        for step, (batch_size, n_id, adjs) in enumerate(valid_loader):      
            adjs = [adj.to(device) for adj in adjs]
            # 获取节点特征
            batch_features = data.x[n_id].to(device)
            # 获取节点标签（对于目标节点）
            batch_labels = data.y[n_id[:batch_size]].to(device)
            temp = model(adjs, batch_features).argmax(1)

            labels.append(batch_labels.cpu().numpy())
            predictions.append(temp.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
    return accuracy
    





def __dname__(p, task_id):
    if p == 0:
        dname = 'task{}.meta.train.support'.format(task_id)
    elif p == 1:
        dname = 'task{}.meta.train.query'.format(task_id)
    elif p == 2:
        dname = 'task{}.meta.test.support'.format(task_id)
    elif p == 3:
        dname = 'task{}.meta.test.query'.format(task_id)
    else:
        raise KeyError

    return dname


def __pos_neg_nodes__(labeled_nodes, node_labels, i: int):
    pos_nodes = labeled_nodes[node_labels[:, i] == 1]
    pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
    neg_nodes = labeled_nodes[node_labels[:, i] == 0]
    neg_nodes = neg_nodes[torch.randperm(neg_nodes.shape[0])]
    return pos_nodes, neg_nodes


def __induced_graph_list_for_graphs__(seeds_list, label, p, num_nodes, potential_nodes, ori_x, same_label_edge_index,
                                      smallest_size, largest_size):
    seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
    induced_graph_list = []
    for seeds in seeds_part_list:

        subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=1, num_nodes=num_nodes,
                                         edge_index=same_label_edge_index, relabel_nodes=True)

        temp_hop = 1
        while len(subset) < smallest_size and temp_hop < 5:
            temp_hop = temp_hop + 1
            subset, _, _, _ = k_hop_subgraph(node_idx=torch.flatten(seeds), num_hops=temp_hop, num_nodes=num_nodes,
                                             edge_index=same_label_edge_index, relabel_nodes=True)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.from_numpy(np.setdiff1d(potential_nodes.numpy(), subset.numpy()))

            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]

            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            # directly downmsample
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
            subset = torch.unique(torch.cat([torch.flatten(seeds), subset]))

        sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

        x = ori_x[subset]
        graph = Data(x=x, edge_index=sub_edge_index, y=label)
        induced_graph_list.append(graph)

    return induced_graph_list


def GPPT_evaluate(model, data, nid, batch_size, device,sample_list):
    valid_loader = NeighborSampler(data.edge_index, node_idx=nid, sizes=sample_list, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=0)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad(): 
        for step, (batch_size, n_id, adjs) in enumerate(valid_loader):      
            adjs = [adj.to(device) for adj in adjs]
            # 获取节点特征
            batch_features = data.x[n_id].to(device)
            # 获取节点标签（对于目标节点）
            batch_labels = data.y[n_id[:batch_size]].to(device)
            temp = model(adjs, batch_features).argmax(1)

            labels.append(batch_labels.cpu().numpy())
            predictions.append(temp.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
    return accuracy
    

