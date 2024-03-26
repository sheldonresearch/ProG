import torch
from prompt_graph.prompt import GPF,GPF_plus
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva
from .task import BaseTask
import time
import warnings
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs
from prompt_graph.evaluation import GpromptEva, AllInOneEva
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
                  self.train_dataset, self.test_dataset, self.val_dataset = split_induced_graphs(self.data, smallest_size=10, largest_size=30)
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
            return loss.item()
      
      def SUPTtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            orth_loss = self.prompt.orthogonal_loss()
            loss += orth_loss
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
            return loss.item()
      
      def GPFTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 

      def AllInOneTrain(self, train_loader):
            #we update answering and prompt alternately.
            
            answer_epoch = 1  # 50
            prompt_epoch = 1  # 50
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss
      
      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center = center_embedding(out, batch.y, self.output_dim)
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader)  
      
      def run(self):
            # for all-in-one and Gprompt we use k-hop subgraph
            train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
            val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
            print("prepare data is finished!")

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
                        
                  elif self.prompt_type == 'All-in-one':
                        loss = self.AllInOneTrain(train_loader)
                        test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                        val_acc, F1 = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)

                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        loss = self.GPFTrain(train_loader)
                        test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
                        val_acc = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.device)
                        
                  elif self.prompt_type =='Gprompt':
                        loss = self.GpromptTrain(train_loader)
                        test_acc = GpromptEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
                        val_acc = GpromptEva(val_loader, self.gnn, self.prompt, self.answering, self.device)
                  
                  if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, time.time() - t0, loss, val_acc, test_acc))
            print(f'Final Test: {final_test_acc:.4f}')
         
            print("Node Task completed")


