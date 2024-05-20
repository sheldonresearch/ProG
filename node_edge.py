import dgl
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import numpy as np
from torch_geometric.utils import subgraph
import pickle
import os
def split_graph(data, split_ratio=0.5):
    num_nodes = data.num_nodes

    # 随机选择一半的节点
    indices = torch.randperm(num_nodes)
    split_point = int(num_nodes * split_ratio)
    nodes1 = indices[:split_point]
    nodes2 = indices[split_point:]

    # 创建掩码来区分子图中的节点
    mask1 = torch.zeros(num_nodes, dtype=torch.bool)
    mask2 = torch.zeros(num_nodes, dtype=torch.bool)
    mask1[nodes1] = True
    mask2[nodes2] = True

    # 映射旧的节点索引到新的索引
    new_index1 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes1)}
    new_index2 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes2)}

    # 筛选并映射edge_index到新的节点索引
    edge_index1 = data.edge_index[:, mask1[data.edge_index[0]] & mask1[data.edge_index[1]]]
    edge_index2 = data.edge_index[:, mask2[data.edge_index[0]] & mask2[data.edge_index[1]]]

    edge_index1 = torch.tensor([[new_index1[idx.item()] for idx in edge_index1[0]],
                                [new_index1[idx.item()] for idx in edge_index1[1]]], dtype=torch.long)
    edge_index2 = torch.tensor([[new_index2[idx.item()] for idx in edge_index2[0]],
                                [new_index2[idx.item()] for idx in edge_index2[1]]], dtype=torch.long)

    data1 = Data(
        x=data.x[mask1],
        edge_index=edge_index1,
        edge_attr=data.edge_attr[mask1[data.edge_index[0]] & mask1[data.edge_index[1]]] if data.edge_attr is not None else None,
        y=data.y[mask1] if data.y is not None else None
    )

    data2 = Data(
        x=data.x[mask2],
        edge_index=edge_index2,
        edge_attr=data.edge_attr[mask2[data.edge_index[0]] & mask2[data.edge_index[1]]] if data.edge_attr is not None else None,
        y=data.y[mask2] if data.y is not None else None
    )

    return data1, data2

def induced_graphs(dir, data, device, smallest_size=1, largest_size=5):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 1

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
        if index%500 == 0:
            print(index)


    dir_path = dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_graph_min'+ str(smallest_size) +'_max'+str(largest_size)+'.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        # pickle.dump(induced_graph_list, f) 
        pickle.dump(induced_graph_list, f)
        print("induced graph data has been write into " + file_path)
    return induced_graph_list



def induced_graphs_from_edges(dir, data, device, smallest_size=1, largest_size=5):
    induced_graph_list = []

    edge_index = data.edge_index
    edge_labels = data.edge_attr

    for edge_id in range(edge_index.size(1)):
        src_node = edge_index[0, edge_id].item()
        tgt_node = edge_index[1, edge_id].item()
        current_label = edge_labels[edge_id].item()

        current_hop = 1

        subset, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=current_hop,
                                         edge_index=edge_index, relabel_nodes=True)
        subset_tgt, _, _, _ = k_hop_subgraph(node_idx=tgt_node, num_hops=current_hop,
                                             edge_index=edge_index, relabel_nodes=True)
        subset = torch.unique(torch.cat([subset, subset_tgt]))

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset_src, _, _, _ = k_hop_subgraph(node_idx=src_node, num_hops=current_hop,
                                                 edge_index=edge_index)
            subset_tgt, _, _, _ = k_hop_subgraph(node_idx=tgt_node, num_hops=current_hop,
                                                 edge_index=edge_index)
            subset = torch.unique(torch.cat([subset_src, subset_tgt]))

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            candidate_nodes = torch.arange(data.x.size(0))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), candidate_nodes])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size]
            subset = torch.unique(torch.cat([torch.LongTensor([src_node, tgt_node]).to(device), subset]))

        sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=torch.tensor([current_label], dtype=torch.long))
        induced_graph_list.append(induced_graph)
        if edge_id%1000==0:
            print(edge_id)
    
    dir_path = dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_edge_graph_min'+ str(smallest_size) +'_max'+str(largest_size)+'.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        # pickle.dump(induced_graph_list, f) 
        pickle.dump(induced_graph_list, f)
        print("induced graph data has been write into " + file_path)

    return induced_graph_list

a=0
device  = torch.device('cuda:'+str(a))


# file_path = 'UGAD_lyq/datasets-edge/amazon-els'
# input_dim=25
# output_dim=2
# dataset_name = 'amazon'

# file_path = 'UGAD_lyq/datasets-edge/questions-els'
# input_dim= 301
# output_dim=2
# dataset_name = 'questions'                                                                                         

# file_path = 'UGAD_lyq/datasets-edge/reddit-els'
# input_dim= 64
# output_dim=2
# dataset_name = 'reddit'     


# file_path = 'UGAD_lyq/datasets-edge/weibo-els'
# input_dim= 400
# output_dim=2
# dataset_name = 'weibo'    

file_path = 'UGAD_lyq/datasets-edge/yelp-els'
input_dim= 32
output_dim=2
dataset_name = 'yelp'    


# file_path = 'UGAD_lyq/datasets-edge/tolokers-els'
# input_dim= 10
# output_dim=2
# dataset_name = 'tolokers'    

graphs, labels = dgl.load_graphs(file_path)



dgl_graph= graphs[0]
edge_index = torch.tensor([dgl_graph.edges()[0].numpy(), dgl_graph.edges()[1].numpy()], dtype=torch.long)
x = torch.tensor(dgl_graph.ndata['feature'].numpy(), dtype=torch.float) 
edge_label = torch.tensor(dgl_graph.edata['edge_label'].numpy(), dtype=torch.float)
y = dgl_graph.ndata['node_label']
print(x.shape)
graph = Data(x=x, edge_index=edge_index, y = y, edge_attr=edge_label)

pretrain_graph_list = []
# from prompt_graph.data import NodePretrain
# pretrain_graph_list = NodePretrain(graph, num_parts=200)
# from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, PrePrompt, DGI, GraphMAE
# pt = GraphMAE(pretrain_graph_list, input_dim, gnn_type = 'GCN', dataset_name = dataset_name, hid_dim = 128, gln = 2, num_epoch=1000,
#                   mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2)
# pt.pretrain()
# # 预训练结束


train_node_edge_graph_list = []
# 使用 train_test_split 按照6:4比例划分数据
train_graph, test_graph = split_graph(graph, split_ratio=0.4)
train_graph.to(device)
test_graph.to(device)
train_node_graph_list = induced_graphs('./lyq/'+dataset_name+'train',train_graph, device)
train_edge_graph_list = induced_graphs_from_edges('./lyq/'+dataset_name+'train',train_graph, device)
for g in train_node_graph_list:
    train_node_edge_graph_list.append(g)
for g in train_edge_graph_list:
    train_node_edge_graph_list.append(g)
test_edge_graph_list = induced_graphs_from_edges('./lyq/'+dataset_name+'test',test_graph, device)
test_node_graph_list = induced_graphs('./lyq/'+dataset_name+'test', test_graph, device)

dataset1= train_node_edge_graph_list, test_node_graph_list
dataset2= train_node_edge_graph_list, test_edge_graph_list
dataset3 = train_node_graph_list, test_edge_graph_list
dataset4 = train_edge_graph_list, test_node_graph_list



from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
import random
import numpy as np
import os
import pandas as pd


args = get_args()
seed_everything(args.seed)


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
args.shot_num = 0
args.epochs = 10

import pandas as pd
import random

results = []
data_list = [dataset1, dataset2, dataset3, dataset4]
for args.prompt_type in ['All-in-one', 'Gprompt']:
    for idx, dataset in enumerate(data_list):
        
        # tasker = GraphTask(
        #     pre_train_model_path='Experiment/pre_trained_model/' + dataset_name + '/GraphMAE.GCN.128hidden_dim.pth',
        #     dataset_name=dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, 
        #     prompt_type=args.prompt_type, epochs=args.epochs, shot_num=args.shot_num, device=3, 
        #     lr=params['learning_rate'], wd=params['weight_decay'], batch_size=int(params['batch_size']), 
        #     dataset=dataset, input_dim=input_dim, output_dim=output_dim
        # )
        tasker = GraphTask(
            dataset_name=dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, 
            prompt_type=args.prompt_type, epochs=args.epochs, shot_num=args.shot_num, device=a, 
            lr=0.01, wd=0, batch_size=4096, 
            dataset=dataset, input_dim=input_dim, output_dim=output_dim
        )
        pre_train_type = tasker.pre_train_type

        mean_test_acc, mean_f1, mean_roc, mean_prc = tasker.run()
   
        print('prompt_type', args.prompt_type)
        print("After searching, Final F1 {:.4f}".format(mean_f1)) 
        print("After searching, Final AUROC {:.4f}".format(mean_roc) )
        print("After searching, Final AUPRC {:.4f}".format(mean_prc))
    
        
        results.append({
            'dataset_name':dataset_name,
            'prompt_type': args.prompt_type,
            'dataset': 'dataset'+str(idx),
            'mean_f1': mean_f1,
            'mean_roc': mean_roc,
            'mean_prc': mean_prc,
        })

# Save results to an Excel file
df = pd.DataFrame(results)
df.to_excel(dataset_name+'node_edge_results.xlsx', index=False)




