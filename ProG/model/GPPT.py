import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv,GINConv,SAGEConv
from torch_geometric.nn import GraphConv as GConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.cluster import KMeans

class GPPT(nn.Module):
    def __init__(self, in_feats, n_hidden=128, n_classes=None, n_layers=2, activation = F.relu, dropout=0.5, center_num=3, device = None):
        super(GPPT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes=n_classes
        self.center_num=center_num
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden))

        self.prompt=nn.Linear(n_hidden,self.center_num,bias=False).to(device)
        
        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2*n_hidden,n_classes,bias=False))
        self.pp.to(device)
        
    def model_to_array(self,args):
        s_dict = torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in np.arange(1, len(keys), 1):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res
    def array_to_model(self, args):
        arr=self.model_to_array(args)
        m_m=torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')#+str(args.gpu))
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():
            length = torch.prod(torch.tensor(param.shape))
            s_dict[name] = arr[indice:indice + length].view(param.shape)
            indice = indice + length
        self.load_state_dict(s_dict)
    
    def load_parameters(self, args):
        self.args=args
        self.array_to_model(args)

    def weigth_init(self, x, edge_index, label,index):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)
        
        features=h[index]
        labels=label[index.long()]
        cluster = KMeans(n_clusters=self.center_num, n_init=10, random_state=0).fit(features.detach().cpu())
        
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data = temp.clone().detach()
        

        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0)
        for i in range(self.center_num):
            self.pp[i].weight.data = temp.clone().detach()
        
    
    def update_prompt_weight(self,h):
        cluster = KMeans(n_clusters=self.center_num, n_init=10, random_state=0).fit(h.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data = temp.clone().detach()

    def get_mul_prompt(self):
        pros=[]
        for name,param in self.named_parameters():
            if name.startswith('pp.'):
                pros.append(param)
        return pros
        
    def get_prompt(self):
        for name,param in self.named_parameters():
            if name.startswith('prompt.weight'):
                pro=param
        return pro
    
    def get_mid_h(self):
        return self.fea

    def forward(self, x, edge_index):

        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if l != len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        x = self.activation(x)
        
        self.fea=x 
        h = x
        out=self.prompt(h)
        index=torch.argmax(out, dim=1)
        out=torch.FloatTensor(h.shape[0],self.n_classes)
        for i in range(self.center_num):
            out[index==i]=self.pp[i](h[index==i])
        return out