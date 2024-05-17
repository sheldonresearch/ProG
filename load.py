import dgl
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import numpy as np

def induced_graphs(data, smallest_size=1, largest_size=10):

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
                subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

            x = data.x[subset]

            induced_graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor([current_label], dtype=torch.long))
            induced_graph_list.append(induced_graph)
            # print(index)
    return induced_graph_list




# 定义文件路径
file_path = 'UGAD_lyq/datasets-graph/mutag0'

# 加载图数据
graphs, labels = dgl.load_graphs(file_path)

# 将labels转换为列表
labels = labels['glabel'].tolist()

# 使用 train_test_split 按照6:4比例划分数据
train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.4, random_state=42)

print('Number of training graphs:', len(train_graphs))
print('Number of testing graphs:', len(test_graphs))


train_graph_list = []
for i in range(len(train_graphs)):
    dgl_graph, label = train_graphs[i], train_labels[i]

    edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
    x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 
    y = torch.tensor([label], dtype=torch.long)
    node_labels = torch.tensor(dgl_graph.ndata['node_label'].numpy(), dtype=torch.long) 

    pyg_graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), node_labels=node_labels)
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    induced_graph_list = induced_graphs(pyg_graph)
    train_graph_list.append(graph)
    for g in induced_graph_list:
        train_graph_list.append(g)

test_graph_list = []
test_induced_graph_list = []
for i in range(len(test_graphs)):
    dgl_graph, label = test_graphs[i], test_labels[i]

    edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
    x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 
    y = torch.tensor([label], dtype=torch.long)
    node_labels = torch.tensor(dgl_graph.ndata['node_label'].numpy(), dtype=torch.long) 

    pyg_graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), node_labels=node_labels)
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    induced_graph_list = induced_graphs(pyg_graph)
    test_graph_list.append(graph)
    for g in induced_graph_list:
        test_induced_graph_list.append(g)




from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import pandas as pd


args = get_args()
seed_everything(args.seed)

param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':  10 ** np.linspace(-5, -6, 1000),
    'batch_size': np.linspace(4096, 4096, 1),
}


num_iter=1
if args.prompt_type in['MultiGprompt','GPPT']:
    num_iter = 1
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
args.epochs = 10
input_dim=14
output_dim=2
dataset = train_graph_list, test_graph_list
dataset2 = train_graph_list, test_induced_graph_list


for args.prompt_type in['All-in-one', 'Gprompt']:
    for dataset in [dataset,dataset2]:
        for _ in range(num_iter):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            
            tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                                dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                                shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                                batch_size = int(params['batch_size']), dataset = dataset, input_dim = input_dim, output_dim = output_dim,)
            pre_train_type = tasker.pre_train_type

            # 返回平均损失
            avg_best_loss, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc = tasker.run()
            print(f"For {_}th searching, Tested Params: {params}, Avg Best Loss: {avg_best_loss}")

            print('prompt_type',args.prompt_type)
            print("After searching, Final Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc)) 
            print("After searching, Final F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1)) 
            print("After searching, Final AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc)) 
            print("After searching, Final AUROC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc)) 
            print('best_params ', best_params)
            print('best_loss ',best_loss)




