from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args

args = get_args()
seed_everything(args.seed)

# args.prompt_type = 'GPPT'

# args.prompt_type = 'All-in-one'

# args.dataset_name = 'Texas'
# args.pre_train_model_path = './Experiment/pre_trained_model/CiteSeer/Edgepred_GPPT.GCN.128hidden_dim.pth'


# args.task = 'NodeTask'
# args.batch_size = 10
# # # args.epochs = 10
# args.dataset_name = 'ogbn-arxiv'

# args.prompt_type = 'None'
# args.pre_train_model_path = './multigprompt_model/cora.multigprompt.GCL.128hidden_dim.pth'



if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer,
                    gnn_type = args.gnn_type, prompt_type = args.prompt_type,
                    epochs = args.epochs, shot_num = args.shot_num, device=args.device, batch_size = args.batch_size, lr =0.01)
    
    tasker.run()


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=args.device)
    tasker.run()