import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random

from prompt import GNN
from utils import gen_ran_output,load_data4pretrain,mkdir, graph_views

class GraphCL(torch.nn.Module):

    def __init__(self, gnn, hid_dim=16):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hid_dim, hid_dim))

    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean() + 10
        return loss


class PreTrain(torch.nn.Module):
    def __init__(self, pretext="GraphCL", gnn_type='TransformerConv',
                 input_dim=None, hid_dim=None, gln=2):
        super(PreTrain, self).__init__()
        self.pretext = pretext
        self.gnn_type=gnn_type

        self.gnn = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gln, pool=None,
                       gnn_type=gnn_type)

        if pretext in ['GraphCL', 'SimGRACE']:
            self.model = GraphCL(self.gnn, hid_dim=hid_dim)
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def get_loader(self, graph_list, batch_size,
                   aug1=None, aug2=None, aug_ratio=None, pretext="GraphCL"):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        if pretext == 'GraphCL':
            shuffle(graph_list)
            if aug1 is None:
                aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug2 is None:
                aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
            if aug_ratio is None:
                aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

            print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

            view_list_1 = []
            view_list_2 = []
            for g in graph_list:
                view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                view_list_1.append(view_g)
                view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                view_list_2.append(view_g)

            loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                 num_workers=1)  # you must set shuffle=False !
            loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                 num_workers=1)  # you must set shuffle=False !

            return loader1, loader2
        elif pretext == 'SimGRACE':
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
            return loader, None  # if pretext==SimGRACE, loader2 is None
        else:
            raise ValueError("pretext should be GraphCL, SimGRACE")

    def train_simgrace(self, model, loader, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            x2 = gen_ran_output(data, model) 
            x1 = model.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(device), requires_grad=False)
            loss = model.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train_graphcl(self, model, loader1, loader2, optimizer):
        model.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = model.forward_cl(batch1.x.to(device), batch1.edge_index.to(device), batch1.batch.to(device))
            x2 = model.forward_cl(batch2.x.to(device), batch2.edge_index.to(device), batch2.batch.to(device))
            loss = model.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def train(self, dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01,
              decay=0.0001, epochs=100):

        loader1, loader2 = self.get_loader(graph_list, batch_size, aug1=aug1, aug2=aug2,
                                           pretext=self.pretext)
        # print('start training {} | {} | {}...'.format(dataname, pre_train_method, gnn_type))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            if self.pretext == 'GraphCL':
                train_loss = self.train_graphcl(self.model, loader1, loader2, optimizer)
            elif self.pretext == 'SimGRACE':
                train_loss = self.train_simgrace(self.model, loader1, optimizer)
            else:
                raise ValueError("pretext should be GraphCL, SimGRACE")

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                torch.save(self.model.gnn.state_dict(),
                           "./pre_trained_gnn/{}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type))
                # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
                # only selected pre-trained models will be moved into (1) so that we can keep reproduction
                print("+++model saved ! {}.{}.{}.pth".format(dataname, self.pretext, self.gnn_type))



if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    # device = torch.device('cpu')

    mkdir('./pre_trained_gnn/')
    # do not use '../pre_trained_gnn/' because hope there should be two folders: (1) '../pre_trained_gnn/'  and (2) './pre_trained_gnn/'
    # only selected pre-trained models will be moved into (1) so that we can keep reproduction

    # pretext = 'GraphCL' 
    pretext = 'SimGRACE' 
    gnn_type = 'TransformerConv'  
    # gnn_type = 'GAT'
    # gnn_type = 'GCN'
    dataname, num_parts = 'CiteSeer', 200
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)
    pt.model.to(device) 
    pt.train(dataname, graph_list, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None,lr=0.01, decay=0.0001,epochs=100)
