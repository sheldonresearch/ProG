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
from torch_geometric.utils import structured_negative_sampling




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
    

class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0,i].cpu().item()) + "," + str(data.edge_index[1,i].cpu().item()) for i in range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2,5*num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5*num_edges):
            node1 = redandunt_sample[0,i].cpu().item()
            node2 = redandunt_sample[1,i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges/2:
                break

        data.negative_edge_index = redandunt_sample[:,sampled_ind]
        
        return data



def prepare_structured_link_prediction_data(graph_data: Data):
    r"""Prepare structured <i,k,j> format link prediction data"""
    node_idx = torch.LongTensor([i for i in range(graph_data.num_nodes)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([graph_data.edge_index, self_loop], dim=1)
    v, a, b = structured_negative_sampling(edge_index, graph_data.num_nodes)
    data = torch.stack([v, a, b], dim=1)

    # (num_edge, 3)
    #   for each entry (i,j,k) in data, (i,j) is a positive sample while (i,k) forms a negative sample
    return data




def convert_edge_index_to_sparse_matrix(edge_index: torch.LongTensor, num_node: int):
    node_idx = torch.LongTensor([i for i in range(num_node)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    sp_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size((num_node, num_node)))

    return sp_adj