import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', type = str)
    parser.add_argument('--dataset_name', type=str, default='Cora',help='Choose the dataset of pretrainor downstream task')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='Embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="GCN")
    # parser.add_argument('--tuning_type', type=str, default="gpf", help='\'gpf\' for GPF and \'gpf-plus\' for GPF-plus in the paper')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 0, help='Number of workers for dataset loading')
    parser.add_argument('--eval_train', type=int, default = 1, help='Evaluating training or not')
    parser.add_argument('--split', type=str, default = "species", help='The way of dataset split(e.g., \'species\' for bio data)')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')
    parser.add_argument('--shot_number', type=int, default = 50, help='Number of shots')

    args = parser.parse_args()
    return args