import torch
from ProG.prompt import GPF,GPF_plus
from ProG.utils import constraint
from .task import BaseTask
import time
import warnings
from ProG.data import load4node
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
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            self.data.to(self.device)
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes

      def train(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            if self.prompt_type in ['gpf', 'gpf-plus']:
                  data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None, prompt = self.prompt, prompt_type=self.prompt_type) 
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


      def test(self, data, mask):
            self.gnn.eval()
            if self.prompt_type in ['gpf', 'gpf-plus']:
                  data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None, prompt = self.prompt, prompt_type = self.prompt_type)
            out = self.answering(out)
            pred = out.argmax(dim=1) 
            correct = pred[mask] == data.y[mask]  
            acc = int(correct.sum()) / int(mask.sum())  
            return acc
      
      def GPPTtest(self, data, mask):
            # self.gnn.eval()
            self.prompt.eval()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            pred = out.argmax(dim=1)  
            correct = pred[mask] == data.y[mask]  
            acc = int(correct.sum()) / int(mask.sum())  
            return acc
      
      def run(self):
            best_val_acc = final_test_acc = 0
            for epoch in range(1, self.epochs):
                  t0 = time.time()
                  if self.prompt_type == 'GPPT':
                        loss = self.GPPTtrain(self.data)
                        val_acc = self.GPPTtest(self.data, self.data.val_mask)
                        test_acc = self.GPPTtest(self.data, self.data.test_mask)
                  else:
                        loss = self.train(self.data)
                        val_acc = self.test(self.data, self.data.val_mask)
                        test_acc = self.test(self.data, self.data.test_mask)
                  if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, time.time() - t0, loss.item(), val_acc, test_acc))
            print(f'Final Test: {final_test_acc:.4f}')
         
            print("Task completed")


