# Graph Task
cd ..

# GCN
python downstream_task.py --pre_train_model_path 'None' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# GraphCL + GCN
python downstream_task.py --pre_train_model_path './pre_trained_gnn/MUTAG.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# GraphCL + All-in-one + GCN
python downstream_task.py --pre_train_model_path './pre_trained_gnn/MUTAG.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'All-in-one' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# GraphCL + Gprompt + GCN
python downstream_task.py --pre_train_model_path './pre_trained_gnn/MUTAG.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
python downstream_task.py --pre_train_model_path './pre_trained_gnn/ENZYMES.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'ENZYMES' --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# GraphCL + GPF + GCN
python downstream_task.py --pre_train_model_path './pre_trained_gnn/MUTAG.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'GPF' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# GraphCL + GPF-plus + GCN
python downstream_task.py --pre_train_model_path './pre_trained_gnn/MUTAG.GraphCL.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'GPF-plus' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

