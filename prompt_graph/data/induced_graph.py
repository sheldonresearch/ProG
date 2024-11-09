# from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data, Batch
import random
import os
from prompt_graph.utils import mkdir
from random import shuffle
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
import pickle

def induced_graphs(data, smallest_size=10, largest_size=30):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)
        # print(index)
    return induced_graph_list



def split_induced_graphs(data, dir_path, device, smallest_size=10, largest_size=30):

    induced_graph_list = []
    saved_graph_list = []
    from copy import deepcopy
    
    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        subset = subset


        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label))   # Test data may leak
            pos_nodes = pos_nodes.to('cpu')
            subset = subset.to('cpu')
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]).to(device), torch.flatten(subset)]))

        subset = subset.to(device)
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
        sub_edge_index = sub_edge_index.to(device)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index = index)
        saved_graph_list.append(deepcopy(induced_graph).to('cpu'))
        induced_graph_list.append(induced_graph)
        if index%500 == 0:
            print(index)


    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_graph_min'+ str(smallest_size) +'_max'+str(largest_size)+'.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        # pickle.dump(induced_graph_list, f) 
        pickle.dump(saved_graph_list, f)
        print("induced graph data has been write into " + file_path)


