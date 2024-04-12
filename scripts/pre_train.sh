
cd ..

python pre_train.py --task Edgepred_Gprompt --dataset_name 'PROTEIN' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
python pre_train.py --task Edgepred_GPPT --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
python pre_train.py --task SimGRACE --dataset_name 'PROTEINS' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
python pre_train.py --task GraphCL --dataset_name 'PROTEINS' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
python pre_train.py --task GraphCL --dataset_name 'COX2' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5