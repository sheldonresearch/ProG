import torch
from ProG.prompt import GPF,GPF_plus
from ProG.utils import constraint
from ProG.evaluation import GPPTEva, GNNNodeEva,GPFNodeEva
from .task import BaseTask
import time
import warnings
from ProG.data import load4node, induced_graphs, graph_split
warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load_data()
            self.initialize_gnn()
            self.initialize_prompt()
            self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device)
            self.initialize_optimizer()
      
      def load_data(self):
            if self.prompt_type in ['All-in-one', 'Gprompt']:
                  self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
                  self.data.to('cpu')
                  self.input_dim = self.dataset.num_features
                  self.output_dim = self.dataset.num_classes
                  self.train_dataset, self.test_dataset, self.val_dataset = induced_graphs(self.data, smallest_size=10, largest_size=30)
                  # for g in graph_list:
                  #       g.to(self.device)
 

            else:
                  self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
                  self.data.to(self.device)
                  self.input_dim = self.dataset.num_features
                  self.output_dim = self.dataset.num_classes
            

      def train(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            loss.backward()  
            self.optimizer.step()  
            return loss
      
      def GPFtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            loss.backward()  
            self.optimizer.step()  
            return loss
            
      def GPPTtrain(self, data):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
            return loss


      
      def run(self):
            best_val_acc = final_test_acc = 0
            for epoch in range(1, self.epochs):
                  t0 = time.time()
                  if self.prompt_type == 'None':
                        loss = self.train(self.data)
                        val_acc = GNNNodeEva(self.data, self.data.val_mask, self.gnn, self.answering)
                        test_acc = GNNNodeEva(self.data, self.data.test_mask, self.gnn, self.answering)
                  elif self.prompt_type == 'GPPT':
                        loss = self.GPPTtrain(self.data)
                        val_acc = GPPTEva(self.data, self.data.val_mask, self.gnn, self.prompt)
                        test_acc = GPPTEva(self.data, self.data.test_mask, self.gnn, self.prompt)
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        loss = self.GPFtrain(self.data)
                        val_acc = GPFNodeEva(self.data, self.data.val_mask, self.gnn, self.prompt, self.answering)
                        test_acc = GPFNodeEva(self.data, self.data.test_mask, self.gnn, self.prompt, self.answering)
                  # elif self.prompt_type == 'All-in-one':
                  # elif self.prompt_type =='Gprompt':
                  
                  if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, time.time() - t0, loss.item(), val_acc, test_acc))
            print(f'Final Test: {final_test_acc:.4f}')
         
            print("Task completed")


