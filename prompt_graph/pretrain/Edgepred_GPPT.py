import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from prompt_graph.data import load4link_prediction_multi_graph, load4link_prediction_single_graph
from torch.optim import Adam
import time
from .base import PreTrain


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
        
        elif self.dataset_name in  ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY'] :
            graph_list, self.input_dim, self.output_dim = load4link_prediction_multi_graph(self.dataset_name) 
              # 多图数据集的处理逻辑
      
            # 对每个图进行处理，创建一个列表来保存处理后的图
            processed_data_list = []
            for data, edge_label, edge_index in graph_list:
                edge_index = edge_index.transpose(0, 1)  # 转置边索引以匹配TensorDataset的期望格式
                # 对于每个图，创建一个TensorDataset
                processed_data = TensorDataset(edge_label, edge_index)
                processed_data_list.append(processed_data)
            # 使用ConcatDataset来合并所有图的数据集，然后通过DataLoader进行批处理加载
            # 注意：这里假设每个图都被视为一个独立的批次，如果需要不同的处理，可以相应地调整
            concatenated_dataset = torch.utils.data.ConcatDataset(processed_data_list)
            return DataLoader(concatenated_dataset, batch_size=1, shuffle=True)  # 每个'批次'包含一个图

      

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
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()
            print(f"[Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")
            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                torch.save(self.gnn.state_dict(),
                           "../ProG/pre_trained_gnn/{}.{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))

