from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args,graph_views,permute_edges
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.utils import from_networkx
import networkx as nx
import torch_geometric.utils as utils
from torch_geometric.data import Data
import copy
from torch import nn, optim
args = get_args()
seed_everything(args.seed)

args.prompt_type ='Gprompt'

def generate_random_graph(num_nodes, num_edges, feature_dim):

    x = torch.rand((num_nodes, feature_dim))

    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 确保生成的是一个合法的边索引矩阵（无重复边，无自环）
    edge_index = utils.to_undirected(edge_index, num_nodes)
    data = Data(x=x, edge_index=edge_index)
    return data


def initialize_optimizer(prompt_type, lr, wd):
    if prompt_type == 'All-in-one':
        optimizer = optim.Adam( prompt.parameters(), lr=1e-6, weight_decay= wd)
    elif prompt_type in ['GPF', 'GPF-plus']:
        optimizer = optim.Adam(prompt.parameters(), lr=lr, weight_decay=wd)
    elif prompt_type in ['Gprompt']:
        optimizer = optim.Adam(prompt.parameters(), lr=lr, weight_decay=wd)
    elif prompt_type in ['GPPT']:
        optimizer = optim.Adam(prompt.parameters(), lr=2e-3, weight_decay=5e-4)
    return optimizer



# 自定义参数
num_nodes = 2000  # 节点数
num_edges = 500  # 边数
feature_dim = 500  # 节点特征维度

# 生成随机图
data = generate_random_graph(num_nodes, num_edges, feature_dim)
print(data)

random_graph = copy.deepcopy(data)
perturbation_graph = permute_edges(random_graph, aug_ratio=0.1).to(args.device)
print(data)
print(perturbation_graph)
data = data.to(args.device)

tasker = GraphTask(pre_train_model_path='None',dataset_name = 'MUTAG', num_layer = 4, gnn_type = 'GAT', hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                    shot_num = args.shot_num, device=args.device, lr = 0.01, wd = 0,
                    batch_size = 1, dataset = data, input_dim = feature_dim, output_dim = 2)
prompt = tasker.prompt
gnn = tasker.gnn
optimizer = initialize_optimizer(args.prompt_type, 0.01, 0)
loss_fn = torch.nn.MSELoss(reduction="sum")
ori_loss = loss_fn(gnn(data.x,data.edge_index),gnn(perturbation_graph.x, perturbation_graph.edge_index))
# print(gnn(data.x,data.edge_index))
# print(gnn(perturbation_graph.x, perturbation_graph.edge_index))
print(ori_loss)

prompt.train()
total_loss = 0.0 

if args.prompt_type in ['GPF','GPF-plus']:
    for epoch in range(500):
        optimizer.zero_grad() 
        data.x = prompt.add(data.x)
        out1 = gnn(data.x, data.edge_index, batch = None, prompt = tasker.prompt, prompt_type = tasker.prompt_type)

        out2 = gnn(perturbation_graph.x, perturbation_graph.edge_index, batch = None)
        loss = loss_fn(out1, out2)  
        loss.backward(retain_graph=True)  
        optimizer.step()  
        print(loss.item())

if args.prompt_type in ['Gprompt']:
    for epoch in range(200):
        optimizer.zero_grad() 
        out1 = gnn(data.x, data.edge_index)
        out1 = prompt(out1)
        out2 = gnn(perturbation_graph.x, perturbation_graph.edge_index)
        loss = loss_fn(out1, out2)  
        loss.backward(retain_graph=True)  
        optimizer.step()  
        print(loss.item())

if args.prompt_type in ['All-in-one']:
    for epoch in range(500):
        optimizer.zero_grad() 
        prompted_graph = prompt.forward(data)
        # print(prompted_graph)

        out1 = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        out2 = gnn(perturbation_graph.x, perturbation_graph.edge_index)
        loss = loss_fn(out1, out2)  
        loss.backward(retain_graph=True)  
        optimizer.step()  
        print(loss.item())