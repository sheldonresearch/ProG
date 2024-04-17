import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from prompt_graph.data import load4link_prediction_multi_graph, load4link_prediction_single_graph
from torch.optim import Adam
import time
from .base import PreTrain
import os

class Edgepred_GPPT(PreTrain):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)  
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim) 
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.output_dim).to(self.device)  

    def generate_loader_data(self):
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo']:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(self.dataset_name)  
            self.data.to(self.device) 
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            return DataLoader(data, batch_size=64, shuffle=True)
        
        elif self.dataset_name in  ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(self.dataset_name)          
            self.data.to(self.device) 
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            return DataLoader(data, batch_size=64, shuffle=True)
      
    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device

        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.gnn.train()
        for step, (batch_edge_label, batch_edge_index) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)
            
            out = self.gnn(self.data.x, self.data.edge_index)
            node_emb = self.graph_pred_linear(out)
          
            batch_edge_index = batch_edge_index.transpose(0,1)
            batch_pred_log = self.gnn.decode(node_emb,batch_edge_index).view(-1)
            loss = criterion(batch_pred_log, batch_edge_label)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step

    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = 1000000
        patience = 10
        cnt_wait = 0
                 
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()
            print(f"Edgepred_GPPT [Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")
            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break
            print(cnt_wait)
        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        torch.save(self.gnn.state_dict(),
                    "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
        
        print("+++model saved ! {}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))

