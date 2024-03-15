# Graph Task

# GCN
python downstream_task.py --pre_train_path 'None' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# SimGRACE + GCN
python downstream_task.py --pre_train_path './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# SimGRACE + All-in-one + GCN
python downstream_task.py --pre_train_path './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'All-in-one' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# SimGRACE + Gprompt + GCN
python downstream_task.py --pre_train_path './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# SimGRACE + GPF + GCN
python downstream_task.py --pre_train_path './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# SimGRACE + GPF-plus + GCN
python downstream_task.py --pre_train_path './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'GPF-plus' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

