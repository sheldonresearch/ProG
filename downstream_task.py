from ProG.tasker import NodeTask, GraphTask
from ProG.utils import seed_everything
from torchsummary import summary
from ProG.utils import print_model_parameters
from ProG.utils import  get_args
# gnn_type =  'GCN' 'GAT' 'GT' 'GCov' 'GIN' 'GraphSAGE'
# prompt_type = 'All-in-one', 'GPF', GPF-plus', 'GPPT', 'Gprompt'
	

args = get_args()
seed_everything(args.seed)

### NodeTask
if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = 'None', 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num)
    tasker.run()


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = 'None', 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num)
    tasker.run()