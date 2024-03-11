from ProG.tasker import NodeTask, LinkTask, GraphTask
from ProG.prompt import GPF, GPF_plus, GPPTPrompt, GPrompt, LightPrompt
from ProG.utils import seed_everything
from torchsummary import summary
from ProG.utils import print_model_parameters

# gnn_type =  'GCN' 'GAT' 'GT' 'GCov' 'GIN' 'GraphSAGE'
# prompt_type = 'All-in-one', 'GPF', GPF-plus', 'GPPT', 'Gprompt'
	
seed_everything(1)

### NodeTask

# tasker = NodeTask(pre_train_model_path = 'None', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', epochs = 150, shot_num = 5)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPPT', epochs = 150, shot_num = 5)

tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.128hidden_dim.pth', 
                  dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF', epochs = 150, shot_num = 5)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF-plus', epochs = 150, shot_num = 5) bug

# tasker = NodeTask(pre_train_model_path = 'None', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 5)

# tasker = NodeTask(pre_train_model_path = 'None', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 5)

# tasker = NodeTask(pre_train_model_path = 'None', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GAT', prompt_type = 'None', shot_num = 10)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 10)

# tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
#                   dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPPT', shot_num = 1)


### GraphTask



# MUTAG dataset
# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'All-in-one', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = 'None', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF-plus', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'Gprompt', shot_num = 10)

# ENZYMES dataset
# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/ENZYMES.GraphCL.GCN.128hidden_dim.pth', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GCN', prompt_type = 'All-in-one', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = 'None', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GraphSAGE', prompt_type = 'None', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/ENZYMES.GraphCL.GCN.128hidden_dim.pth', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GCN', prompt_type = 'GPF-plus', shot_num = 10)

# tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
#                      dataset_name = 'ENZYMES', num_layer = 3, gnn_type = 'GCN', prompt_type = 'Gprompt', shot_num = 10)


tasker.run()