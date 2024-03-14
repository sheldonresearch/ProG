
cd ..
# python ../pre_rain.py --task Edgepred_Gprompt --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task Edgepred_GPPT --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task SimGRACE --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task GraphCL --dataset_name 'Cora' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

# python ../pre_rain.py --task Edgepred_Gprompt --dataset_name 'CiteSeer' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task Edgepred_GPPT --dataset_name 'CiteSeer' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task SimGRACE --dataset_name 'CiteSeer' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task GraphCL --dataset_name 'CiteSeer' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5

python pre_train.py --task Edgepred_Gprompt --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task Edgepred_GPPT --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task SimGRACE --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
# python ../pre_rain.py --task GraphCL --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
