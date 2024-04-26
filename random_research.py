from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
import random
import numpy as np
import os

args = get_args()
seed_everything(args.seed)

param_grid = {
    'learning_rate': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':  10 ** np.linspace(-5, -6, 1000),
    'batch_size': np.linspace(5, 20, 1),
    # 'hidden_dim': [16, 32, 64, 128, 256],
    # 'dropout_rate': [0, 0.25, 0.5]
}


num_iter=10
best_params = None
best_loss = float('inf')

args.task = 'NodeTask'
# args.prompt_type = 'GPPT'
# args.dataset_name = 'Cora'
args.shot_num = 1

final_acc = 0
final_std = 0
for _ in range(num_iter):
    params = {k: random.choice(v) for k, v in param_grid.items()}
    
    # 假设这是调用你的模型训练函数并返回平均最佳损失
    if args.task == 'NodeTask':
        tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer,
                        gnn_type = args.gnn_type, prompt_type = args.prompt_type,
                        epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = int(params['batch_size']))

    if args.task == 'GraphTask':
        tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                        dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs,
                        shot_num = args.shot_num, device=args.device, lr = params['learning_rate'], wd = params['weight_decay'],
                        batch_size = params['batch_size'])


    avg_best_loss, acc ,std = tasker.run()
    print(f"Tested Params: {params}, Avg Best Loss: {avg_best_loss}")

    if avg_best_loss < best_loss:
        best_loss = avg_best_loss
        best_params = params
        final_acc = acc
        final_std = std
        
            
file_name2 = tasker.gnn_type +"_total_results.txt"
file_path2 = os.path.join('./Experiment/Results/Node_Task/'+str(tasker.shot_num)+'shot/'+ tasker.dataset_name +'/', file_name2)
os.makedirs(os.path.dirname(file_path2), exist_ok=True)
with open(file_path2, 'a') as f:
        
        f.write(" {}_{}_{} Final best | test Accuracy {:.4f}±{:.4f}\n".format(tasker.pre_train_type, tasker.gnn_type, tasker.prompt_type, final_acc, final_std))

print(f"Results saved to {file_path2}") 
print("After searching, Final Accuracy {:.4f}±{:.4f}(std)".format(final_acc, final_std)) 




