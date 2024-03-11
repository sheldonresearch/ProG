from ProG.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
from ProG.utils import seed_everything
from ProG.utils import mkdir

	
seed_everything(1)
mkdir('./pre_trained_gnn/')
pt = SimGRACE(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)
# pt = GraphCL(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)
# pt = Edgepred_GPPT(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)

# pt = Edgepred_GPPT(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)
# pt = Edgepred_Gprompt(dataset_name = 'Cora', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=100)
# pt = GraphCL(dataset_name = 'ENZYMES', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)
# pt = SimGRACE(dataset_name = 'ENZYMES', gnn_type = 'GCN', hid_dim = 128, gln =3, num_epoch=50)

pt.pretrain()

