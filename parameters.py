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
def load_induced_graph(dataset_name, data, device):

    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
seed_everything(args.seed)

param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':  10 ** np.linspace(-5, -6, 1000),
    'batch_size': np.linspace(32, 64, 32),
}
# if args.dataset_name in ['PubMed']:
#      param_grid = {
#     'learning_rate': 10 ** np.linspace(-3, -1, 1000),
#     'weight_decay':  10 ** np.linspace(-5, -6, 1000),
#     'batch_size': np.linspace(128, 512, 200),
#     }
if args.dataset_name in ['ogbn-arxiv','Flickr']:
     param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1),
    'weight_decay':  10 ** np.linspace(-5, -6, 1),
    'batch_size': np.linspace(512, 512, 200),
    }


num_iter=10
print('args.dataset_name', args.dataset_name)
if args.prompt_type in['MultiGprompt','GPPT']:
    print('num_iter = 1')
    num_iter = 1
if args.dataset_name in ['ogbn-arxiv','Flickr']:
    print('num_iter = 1')
    num_iter = 1
best_params = None
best_loss = float('inf')
final_acc_mean = 0
final_acc_std = 0
final_f1_mean = 0
final_f1_std = 0
final_roc_mean = 0
final_roc_std = 0

args.task = 'NodeTask'

args.dataset_name = 'ogbn-arxiv'
# # args.dataset_name = 'Cora'
# # num_iter = 1
args.shot_num = 1
args.device = 1

args.pre_train_model_path='./Experiment/pre_trained_model/ogbn-arxiv/DGI.GCN.128hidden_dim.pth' 

param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':  10 ** np.linspace(-5, -6, 1000),
    'batch_size': np.linspace(32, 64, 32),
}
params = {k: random.choice(v) for k, v in param_grid.items()}
if args.task == 'NodeTask':
    data, input_dim, output_dim = load4node(args.dataset_name)   
    data = data.to(args.device)
    if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
        graphs_list = load_induced_graph(args.dataset_name, data, args.device) 
    else:
        graphs_list = None 
         

if args.task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name)
    
print('num_iter',num_iter)
# for args.prompt_type in[ 'None','GPPT', 'Gprompt', 'All-in-one', 'GPF','GPF-plus', 'MultiGprompt']:
for args.prompt_type in[ 'MultiGprompt']:
    if args.task == 'NodeTask':
        tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = int(params['batch_size']), data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list)


    if args.task == 'GraphTask':
        tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = int(params['batch_size']), dataset = dataset, input_dim = input_dim, output_dim = output_dim)

    if tasker.gnn is not None:
        print(tasker.prompt_type)
        print('gnn parameters')
        print_model_parameters(tasker.gnn)
    if tasker.answering is not None:
        print('answer parameters')
        print_model_parameters(tasker.answering)
    if tasker.prompt is not None:
        print('prompt parameters')
        print_model_parameters(tasker.prompt)