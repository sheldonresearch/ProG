from .base import PreTrain
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
from torch.optim import Adam
import torch
from torch import nn
import time
from prompt_graph.utils import generate_corrupted_graph
from prompt_graph.data import load4node, load4graph, NodePretrain
import os

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class DGI(PreTrain):
    def __init__(self, *args, hid_dim = 16, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        
        
        self.disc = Discriminator(hid_dim).to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.graph_data = self.load_data()
        self.initialize_gnn(self.input_dim, hid_dim)  
        self.optimizer = Adam(self.gnn.parameters(), lr=0.01, weight_decay = 0.0001)

    def load_data(self):
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr']:
            data, dataset = load4node(self.dataset_name)
            self.input_dim = dataset.num_features
        return data
        #     self.graph_list, self.input_dim = NodePretrain(dataname = self.dataset_name, num_parts=200)
        # else:
        #     self.input_dim, _, self.graph_list= load4graph(self.dataset_name, pretrained=True)

    def generate_loader_data(self):
        loader1 = self.graph_data

        # only perturb node indices in transductive setup
        loader2 = generate_corrupted_graph(self.graph_data, "shuffleX")
        return loader1, loader2

    def pretrain_one_epoch(self):
        self.gnn.train()

        device = self.device

        graph_original, graph_corrupted = self.generate_loader_data()
        graph_original.to(device)
        graph_corrupted.to(device)

        pos_z = self.gnn(graph_original.x, graph_original.edge_index)
        neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

        s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)
        # print(pos_z.shape, neg_z.shape, s.shape)

        logits = self.disc(s, pos_z, neg_z)

        lbl_1 = torch.ones((pos_z.shape[0], 1))
        lbl_2 = torch.zeros((neg_z.shape[0], 1))
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

        l = self.loss(logits, lbl)
        l.backward()
        self.optimizer.step()

        accum_loss = float(l.detach().cpu().item())
        return accum_loss
            


    def pretrain(self):
        train_loss_min = 1000000
        for epoch in range(1, self.epochs + 1):
            time0 = time.time()
            self.optimizer.zero_grad()
            train_loss = self.pretrain_one_epoch()
            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, self.epochs , train_loss))

            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'DGI', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'DGI', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
