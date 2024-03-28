from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args


args = get_args()
seed_everything(args.seed)
mkdir('./pre_trained_gnn/')

if args.task == 'SimGRACE':
    pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'GraphCL':
    pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'Edgepred_GPPT':
    pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'Edgepred_Gprompt':
    pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)

pt.pretrain()

