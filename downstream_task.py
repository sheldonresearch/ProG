from ProG.tasker import NodeTask, LinkTask, GraphTask
from ProG.prompt import GPF, GPF_plus, GPPTPrompt, GPrompt, LightPrompt
from ProG.utils import seed_everything


	
seed_everything(1)

tasker = NodeTask(pre_train_model_path = 'None', 
                  dataset_name = 'PubMed', num_layer = 3, gnn_type = 'GAT', prompt_type = 'gppt', shot_num = 1)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 10)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'gppt', shot_num = 1)

# tasker = LinkTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth', 
#                      dataset_name = 'Cora', gnn_type = 'GAT', prompt_type = 'None')

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', gnn_type = 'GCN', prompt_type = 'gpf', shot_num = 50)

# tasker = GraphTask(pre_train_model_path = 'None', 
#                      dataset_name = 'MUTAG', gnn_type = 'GCN', prompt_type = 'Gprompt', shot_num = 20)

# tasker = GraphTask(pre_train_model_path = 'None', 
#                      dataset_name = 'ENZYMES', gnn_type = 'GCN', prompt_type = 'None', shot_num = 50)
tasker.run()