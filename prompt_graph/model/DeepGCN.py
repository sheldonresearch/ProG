import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, GCNConv
class DeeperGCN(torch.nn.Module):
    def __init__(self,input_dim,  hid_dim, num_layer, pool='mean'):
        super().__init__()

        self.node_encoder = Linear(input_dim, hid_dim)
    
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layer + 1):
            conv = GCNConv(hid_dim, hid_dim)
            norm = LayerNorm(hid_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
            if pool == "sum":
                self.pool = global_add_pool
            elif pool == "mean":
                self.pool = global_mean_pool
            elif pool == "max":
                self.pool = global_max_pool
            # elif pool == "attention":
            #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
            else:
                raise ValueError("Invalid graph pooling type.")


    def forward(self, x, edge_index, batch = None, prompt = None, prompt_type = None):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        if batch == None:
            return x
        else:
            if prompt_type == 'Gprompt':
                x = prompt(x)
            graph_emb = self.pool(x, batch.long())
            return graph_emb

