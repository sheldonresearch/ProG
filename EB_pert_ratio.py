from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args,graph_views,permute_edges, drop_nodes,mask_nodes
from prompt_graph.data import load4node,load4graph, split_induced_graphs
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer, DeeperGCN
from prompt_graph.prompt import GPF, GPF_plus, LightPrompt,HeavyPrompt, Gprompt, GPPTPrompt, DiffPoolPrompt, SAGPoolPrompt
import os
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.utils import from_networkx
import networkx as nx
import torch_geometric.utils as utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import FakeDataset
import copy
from torch import nn, optim
args = get_args()
seed_everything(args.seed)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        return rmse
def initialize_optimizer(prompt_type, lr, wd):
    if prompt_type == 'All-in-one':
        optimizer = optim.Adam( prompt.parameters(), lr=lr*2, weight_decay= wd)
    elif prompt_type in ['GPF', 'GPF-plus']:
        optimizer = optim.Adam(prompt.parameters(), lr=lr, weight_decay=wd)
    elif prompt_type in ['Gprompt']:
        optimizer = optim.Adam(prompt.parameters(), lr=lr, weight_decay=wd)
        print('lr', lr)
    elif prompt_type in ['GPPT']:
        optimizer = optim.Adam(prompt.parameters(), lr=2e-3, weight_decay=5e-4)
    return optimizer

def initialize_prompt(prompt_type, input_dim, hid_dim):
    print(prompt_type)
           
    if prompt_type =='All-in-one':
        prompt = HeavyPrompt(token_dim=input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3)
    elif prompt_type == 'GPF':
        prompt = GPF(input_dim)
    elif prompt_type == 'GPF-plus':
        prompt = GPF_plus(input_dim, 20)
    elif prompt_type == 'Gprompt':
        prompt = Gprompt(hid_dim)
    elif prompt_type =='GPPT':
        prompt = GPPTPrompt(hid_dim,1, 1, device = args.device)    
    else:
        raise KeyError(" We don't support this kind of prompt.")
    return prompt

def initialize_gnn(gnn_type, input_dim, hid_dim, num_layer=2):
    if gnn_type == 'GAT':
        gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCN':
        gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphSAGE':
        gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GIN':
        gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GCov':
        gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'GraphTransformer':
        gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    elif gnn_type == 'DeepGCN':
        gnn = DeeperGCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=num_layer)
    else:
        raise ValueError(f"Unsupported GNN type: {gnn_type}")
    return gnn

def get_error_bound(prompt_type, loader1, label, optimizer, prompt, gnn):
    if prompt_type in ['GPF','GPF-plus']:
        for epoch in range(100):
            optimizer.zero_grad() 
            prompt.train()
            for batch in loader1:
                batch = batch.to(args.device)
                batch.x = prompt.add(batch.x)
                out1 = gnn(batch.x, batch.edge_index, batch.batch)

            out2 = label
            loss = loss_fn(out1, out2)  
            loss.backward(retain_graph=True)  
            optimizer.step()  
            # print(loss.item())

    if prompt_type in ['Gprompt']:
        for epoch in range(100):
            optimizer.zero_grad() 
            prompt.train()
            for batch in loader1:
                batch = batch.to(args.device)
                out1 = gnn(batch.x, batch.edge_index, batch.batch, prompt = prompt, prompt_type = prompt_type)

            out2 = label
            loss = loss_fn(out1, out2)  
            loss.backward(retain_graph=True)  
            optimizer.step()  
            # print(loss.item())

    if prompt_type in ['All-in-one']:
        for epoch in range(100):
            optimizer.zero_grad() 
            prompt.train()
            for batch in loader1:
                batch = batch.to(args.device)
                prompted_graph = prompt.forward(batch)
                out1 = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            out2 = label
            loss = loss_fn(out1, out2)  
            loss.backward(retain_graph=True)  
            optimizer.step()  
            # print(loss.item())


    if prompt_type == 'GPPT':        
        for epoch in range(100):
            prompt.train()
            optimizer.zero_grad() 
            for batch in loader1:
                graph_list = batch.to_data_list()        
                for index, graph in enumerate(graph_list):
                    graph=graph.to(args.device)              
                    node_embedding = gnn(graph.x,graph.edge_index)
                    out1 = prompt(node_embedding, graph.edge_index)
            out2 = label
            loss = loss_fn(out1, out2)  
            loss.backward(retain_graph=True)  
            optimizer.step()  
            print(loss.item())
    return loss.item()




# gnn_type= 'GCN'
# nl = 5
# for gnn_type in ['GCN', 'GAT', 'GraphTransformer', 'GraphSAGE']:

for perturbation_type in ['dropN', 'permE', 'maskN']:
    for gnn_type in ['GCN']:
        for nl in [2]:
            file_name = perturbation_type +'_'+ str(nl) +'layer'+gnn_type +"_pert_ratio_Error_bound.xlsx"
            file_path = os.path.join('./Experiment/EB_Results/', file_name)
            if not os.path.exists(file_path):
                # Initialize the DataFrame with appropriate indices and columns
                index = ['ori_error', 'GPF', 'GPF-plus', 'All-in-one', 'Gprompt']
                columns = [str(i/20) for i in range(10, 20)]
                data = pd.DataFrame(index=index, columns=columns)
                data.to_excel(file_path)
            data = pd.read_excel(file_path, index_col=0)

            ratios = [i/20 for i in range(10, 20)]
            for ratio in ratios:
                num_repeats = 5
                cumulative_losses = {prompt_type: 0 for prompt_type in ['GPF', 'GPF-plus', 'All-in-one', 'Gprompt']}
                cumulative_ori_errors = 0
                for a in range(num_repeats):
                    print(str(a+1)+'th calculate...')
                    graphdata = FakeDataset(1,4096,5,500)
                    ori_graph = graphdata[0]
                    print(ori_graph)
                    gnn = initialize_gnn(gnn_type,input_dim=500, hid_dim=128, num_layer=nl).to(args.device)
                
                    random_graph = copy.deepcopy(ori_graph)
                    perturbation_graph = graph_views(random_graph, perturbation_type, aug_ratio=ratio).to(args.device)
                    print(perturbation_graph)
                    loader1 = DataLoader([ori_graph], batch_size=1)
                    loader2 = DataLoader([perturbation_graph], batch_size=1)
                    for prompt_type in ['GPF', 'GPF-plus','All-in-one','Gprompt']:
                    # for prompt_type in ['GPPT']:
                        prompt = initialize_prompt(prompt_type, input_dim=500, hid_dim=128).to(args.device)
                        optimizer = initialize_optimizer(prompt_type, 0.05, 0)
                        loss_fn = RMSELoss()

                        for batch in loader1:
                            batch = batch.to(args.device)
                            graph_embedding1 = gnn(batch.x,batch.edge_index,batch.batch)
                        for batch in loader2:
                            batch = batch.to(args.device)
                            graph_embedding2 = gnn(batch.x,batch.edge_index,batch.batch)

                        def euclidean_distance(x1, x2):
                            return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1)).sum()

                        ori_loss =loss_fn(graph_embedding1, graph_embedding2)
                        cumulative_ori_errors += ori_loss.item()
                        print('ori_error', ori_loss.item())
                        col_name = f"{str(ratio)}"

                        data.at['ori_error', col_name] = f"{cumulative_ori_errors / num_repeats:.8f}"
                        label = graph_embedding2
                        loss = get_error_bound(prompt_type, loader1, label, optimizer, prompt, gnn)
                        cumulative_losses[prompt_type] += loss
                        print(prompt_type, 'Error bound', loss)

                        print('col_name', col_name)
                        data.at[prompt_type, col_name] = f"{cumulative_losses[prompt_type] / num_repeats:.8f}"

            data.to_excel(file_path)

            print("Data saved to "+file_path+" successfully.")