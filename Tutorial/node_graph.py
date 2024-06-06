import dgl
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import numpy as np

def induced_graphs(data, device, smallest_size=1, largest_size=5):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.node_labels[index].item()

        current_hop = 2

        max_node_idx = torch.max(edge_index)
        if max_node_idx > index:
        #     induced_graph = Data(x=data.x[index],  y=current_label)
        #     induced_graph_list.append(induced_graph)

        # else:
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
                subset = torch.unique(torch.cat([torch.LongTensor([index]).to(device), torch.flatten(subset)]))
            subset = subset.to(device)
            sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
            sub_edge_index = sub_edge_index.to(device)
            x = data.x[subset]

            induced_graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor([current_label], dtype=torch.long))
            induced_graph_list.append(induced_graph)
            # print(index)
        # if index%500 == 0:
        #     print(index)
    return induced_graph_list

                          

# 定义文件路径
# file_path = 'UGAD_lyq/datasets-graph/mutag0'
# input_dim=14
# output_dim=2
# dataset_name = 'mutag'

# file_path = 'UGAD_lyq/datasets-graph/bm_mn_dgl'
# input_dim=1
# output_dim=2
# dataset_name = 'bm_mn'                                                                                         

# file_path = 'UGAD_lyq/datasets-graph/bm_ms_dgl'
# input_dim=1
# output_dim=2
# dataset_name = 'bm_ms'    

# file_path = 'UGAD_lyq/datasets-graph/bm_mt_dgl'
# input_dim=1
# output_dim=2
# dataset_name = 'bm_mt'    

file_path = 'UGAD_lyq/datasets-graph/mnist0'
input_dim=5
output_dim=2
dataset_name = 'mnist0'

# file_path = 'UGAD_lyq/datasets-graph/mnist1'
# input_dim=5
# output_dim=2
# dataset_name = 'mnist1'


graphs, labels = dgl.load_graphs(file_path)
a = 4
device  = torch.device('cuda:'+str(a))

pretrain_graph_list = []
for i in range(len(graphs)):
    dgl_graph= graphs[i]

    edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
    x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 

    graph = Data(x=x, edge_index=edge_index, dtype=torch.long).to(device)
    pretrain_graph_list.append(graph)

from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, PrePrompt, DGI, GraphMAE
pt = GraphMAE(pretrain_graph_list, input_dim, gnn_type = 'GCN', dataset_name = dataset_name, hid_dim = 128, gln = 2, num_epoch=1000,
                  mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2)
pt.pretrain()

# 将labels转换为列表
labels = labels['glabel'].tolist()

# 使用 train_test_split 按照6:4比例划分数据
train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.6, random_state=42)

print('Number of training graphs:', len(train_graphs))
print('Number of testing graphs:', len(test_graphs))


train_graph_list = []
train_node_list = []
train_node_graph_list = []
for i in range(len(train_graphs)):
    dgl_graph, label = train_graphs[i], train_labels[i]

    edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
    x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 
    y = torch.tensor([label], dtype=torch.long)
    # print(x.shape,y.shape)
    node_labels = torch.tensor(dgl_graph.ndata['node_label'].numpy(), dtype=torch.long) 

    pyg_graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), node_labels=node_labels).to(device)
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)).to(device)

    train_node_graph_list.append(graph)
    induced_graph_list = induced_graphs(pyg_graph,device)
    graph.to(device)
    train_graph_list.append(graph)
    for g in induced_graph_list:
        g.to(device)
        train_node_list.append(g)
        train_node_graph_list.append(g)
    if i%500==0:
        print(i)


test_graph_list = []
test_node_list = []
for i in range(len(test_graphs)):
    dgl_graph, label = test_graphs[i], test_labels[i]

    edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
    x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 
    y = torch.tensor([label], dtype=torch.long)
    # print(x.shape,y.shape)
    node_labels = torch.tensor(dgl_graph.ndata['node_label'].numpy(), dtype=torch.long) 

    pyg_graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), node_labels=node_labels).to(device)


    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)).to(device)
    induced_graph_list = induced_graphs(pyg_graph,device)
    test_graph_list.append(graph)
    for g in induced_graph_list:
        test_node_list.append(g)
    if i%500==0:
        print(i)

from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
import random
import numpy as np
import os
import pandas as pd


args = get_args()
seed_everything(args.seed)

# param_grid = {
#     'learning_rate': 10 ** np.linspace(-3, -1, 1000),
#     'weight_decay':  10 ** np.linspace(-5, -6, 1000),
#     'batch_size': np.linspace(8192, 4096, 1),
# }


num_iter=1
best_params = None
best_loss = float('inf')
final_acc_mean = 0
final_acc_std = 0
final_f1_mean = 0
final_f1_std = 0
final_roc_mean = 0
final_roc_std = 0
final_prc_mean = 0
final_prc_std = 0

args.task = 'GraphTask'
args.prompt_type = 'All-in-one'
args.shot_num = 0
args.epochs = 100

dataset1 = train_graph_list, test_node_list
dataset2 = train_node_list, test_graph_list
dataset3 = train_node_graph_list, test_node_list
dataset4 = train_node_graph_list, test_graph_list
task = ['graph2node', 'node2graph','all2node','all2graph']

import pandas as pd
import random

results = []
data_list = [dataset1, dataset2, dataset3, dataset4]
for args.prompt_type in ['Gprompt','All-in-one' ]:
    for idx, dataset in enumerate(data_list):
  
        tasker = GraphTask(
            pre_train_model_path='Experiment/pre_trained_model/' + dataset_name + '/GraphMAE.GCN.128hidden_dim.pth',
            dataset_name=dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, 
            prompt_type=args.prompt_type, epochs=args.epochs, shot_num=args.shot_num, device=a, 
            lr=0.1, wd=5e-4, batch_size=8192, 
            dataset=dataset, input_dim=input_dim, output_dim=output_dim
        )
        # tasker = GraphTask(
        #     dataset_name=dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, 
        #     prompt_type=args.prompt_type, epochs=args.epochs, shot_num=args.shot_num, device=a, 
        #     lr=params['learning_rate'], wd=params['weight_decay'], batch_size=int(params['batch_size']), 
        #     dataset=dataset, input_dim=input_dim, output_dim=output_dim
        # )
        pre_train_type = tasker.pre_train_type

        # 返回平均损失
        mean_test_acc, mean_f1, mean_roc, mean_prc = tasker.run()
   
        print('prompt_type', args.prompt_type)
        print("After searching, Final F1 {:.4f}".format(mean_f1)) 
        print("After searching, Final AUROC {:.4f}".format(mean_roc) )
        print("After searching, Final AUPRC {:.4f}".format(mean_prc))
    
        
        results.append({
            'dataset_name':dataset_name,
            'prompt_type': args.prompt_type,
            'dataset': task[idx],
            'mean_f1': mean_f1,
            'mean_roc': mean_roc,
            'mean_prc': mean_prc,
        })

# Save results to an Excel file
df = pd.DataFrame(results)
df.to_excel(dataset_name+'node_graph_results.xlsx', index=False)



