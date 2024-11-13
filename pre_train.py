import argparse
from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, NodePrePrompt, GraphPrePrompt, DGI, GraphMAE
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args
from prompt_graph.data import load4node,load4graph



def get_pretrain_task_delegate(args:argparse.Namespace):
    seed_everything(args.seed)
    if args.pretrain_task == 'SimGRACE':
        pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'GraphCL':
        pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'Edgepred_GPPT':
        pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'Edgepred_Gprompt':
        pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'DGI':
        pt = DGI(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'NodeMultiGprompt':
        nonlinearity = 'prelu'
        pt = NodePrePrompt(args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3, device=args.device)
    elif args.pretrain_task == 'GraphMultiGprompt':
        #TODO: Bugged unknown parameters: graph_list, input_dim, out_dim
        nonlinearity = 'prelu'
        pt = GraphPrePrompt(graph_list, input_dim, out_dim, args.dataset_name, args.hid_dim, nonlinearity,0.9,0.9,0.1,1,0.3, device=args.device)
    elif args.pretrain_task == 'GraphMAE':
        pt = GraphMAE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device,
                    mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2, num_workers=args.num_workers)
    else:
        raise ValueError(f"Unexpected args.pretrain_task type: {args.pretrain_task}.")
    return pt



if __name__ == "__main__":
    args = get_args()
    

    pt = get_pretrain_task_delegate(args=args)
    pt.pretrain()