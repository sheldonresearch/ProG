import wandb
from prompt_graph.tasker import NodeTask_PAR
import math
sweep_config = {
"name": "gcn-sweep",
"method": "random",
"parameters": {
        'batch_size': {
        # integers between 10 and 20
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform',
        'q': 1,
        'min': math.log(10),
        'max': math.log(20),
      },
        "weight_decay": {
            "distribution": "normal",
            "mu": 5e-4,
            "sigma": 1e-5,
        },
        "lr": {
            'distribution': 'uniform',
            "min": 1e-3,
            "max": 1e-2
        }
}
}

# Register the Sweep with W&B
from prompt_graph.utils import  get_args
from prompt_graph.utils import seed_everything
from pprint import pprint



args = get_args()
seed_everything(args.seed)

tasker = NodeTask_PAR(pre_train_model_path = args.pre_train_model_path, 
                dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=args.device)

sweep_id = wandb.sweep(sweep_config, project=tasker.pre_train_type+'+' + tasker.gnn_type +'+'+ tasker.prompt_type)

pprint(sweep_config)
wandb.agent(sweep_id, project=tasker.pre_train_type+'+' + tasker.gnn_type +'+'+ tasker.prompt_type, function=tasker.run, count=10)