import torch
from prompt_graph.prompt import GPF,GPF_plus
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva
from prompt_graph.pretrain import PrePrompt, prompt_pretrain_sample
from .task import BaseTask
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save
from prompt_graph.evaluation import GpromptEva, AllInOneEva
import pickle
import os
import copy
from prompt_graph.utils import process
warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.load_data()
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
            
            # self.create_few_data_folder()         
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
      

      def create_few_data_folder(self):
            # 创建文件夹并保存数据
            for k in range(1, 11):
                  k_shot_folder = './sample_data/'+ self.dataset_name +'/' + str(k) +'_shot'
                  os.makedirs(k_shot_folder, exist_ok=True)
                  
                  for i in range(1, 6):
                        folder = os.path.join(k_shot_folder, str(i))
                        os.makedirs(folder, exist_ok=True)
                        node_sample_and_save(self.data, k, folder, self.output_dim)
                        print(str(k) + ' shot ' + str(i) + ' th is saved!!')

      def load_multigprompt_data(self):
            adj, features, labels, idx_train, idx_val, idx_test = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).cuda()
            self.idx_train = torch.LongTensor(idx_train)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)
            self.idx_val = torch.LongTensor(idx_val)
            self.idx_test = torch.LongTensor(idx_test)

      def load_induced_graph(self):
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            self.data.to('cpu')
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
            file_path = './induced_graph/' + self.dataset_name + '_induced_graph.pkl'
            if os.path.exists(file_path):
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            else:
                  print('Begin split_induced_graphs.')
                  split_induced_graphs(self.dataset_name, self.data, smallest_size=10, largest_size=30)
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            return graphs_list

      
      def load_data(self):
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            self.data.to(self.device)
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
      
      def train(self, data, train_idx):
            self.gnn.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
      
      def GPPTtrain(self, data, train_idx):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
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

            if self.prompt_type != 'MultiGprompt':
                  test_accs = []
                  
                  for i in range(1, 6):
                        idx_train = torch.load("./sample_data/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                        print('idx_train',idx_train)
                        train_lbls = torch.load("./sample_data/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                        print("true",i,train_lbls)

                        idx_test = torch.load("./sample_data/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                        test_lbls = torch.load("./sample_data/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                        
                        # for all-in-one and Gprompt we use k-hop subgraph
                        if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                              graphs_list = self.load_induced_graph()
                              train_graphs = []
                              test_graphs = []
                              
                              for graph in graphs_list:                              
                                    if graph.index in idx_train:
                                          train_graphs.append(graph)
                                    elif graph.index in idx_test:
                                          test_graphs.append(graph)

                                    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
                                    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
                                    print("prepare induce graph data is finished!")

                        patience = 20
                        best = 1e9
                        cnt_wait = 0
                        
                        for epoch in range(1, self.epochs):

                              t0 = time.time()
                              if self.prompt_type == 'None':
                                    loss = self.train(self.data, idx_train)                             
                              elif self.prompt_type == 'GPPT':
                                    loss = self.GPPTtrain(self.data, idx_train)                
                              elif self.prompt_type == 'All-in-one':
                                    loss = self.AllInOneTrain(train_loader, idx_train)                           
                              elif self.prompt_type in ['GPF', 'GPF-plus']:
                                    loss = self.GPFTrain(train_loader, idx_train)                                                          
                              elif self.prompt_type =='Gprompt':
                                    loss = self.GpromptTrain(train_loader, idx_train)

                              if loss < best:
                                    best = loss
                                    # best_t = epoch
                                    cnt_wait = 0
                                    # torch.save(model.state_dict(), args.save_name)
                              else:
                                    cnt_wait += 1
                                    if cnt_wait == patience:
                                          print('-' * 100)
                                          print('Early stopping at '+str(epoch) +' eopch!')
                                          break
                              print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                 
                        if self.prompt_type == 'None':
                              test_acc = GNNNodeEva(self.data, idx_test, self.gnn, self.answering)                           
                        elif self.prompt_type == 'GPPT':
                              test_acc = GPPTEva(self.data, idx_test, self.gnn, self.prompt)                
                        elif self.prompt_type == 'All-in-one':
                              test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)                                           
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)                                                         
                        elif self.prompt_type =='Gprompt':
                              test_acc = GpromptEva(test_loader, self.gnn, self.prompt, self.answering, self.device)

                        print("test accuracy {:.4f} ".format(test_acc))                        
                        test_accs.append(test_acc)
               

                  mean_test_acc = np.mean(test_accs)
                  std_test_acc = np.std(test_accs)    
                  print(" Final best | test Accuracy {:.4f} | std {:.4f} ".format(mean_test_acc, std_test_acc))         
                  
            elif self.prompt_type == 'MultiGprompt':
                  # embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                  embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)

                  preval_embs = embeds[0, self.idx_val]
                  test_embs = embeds[0, self.idx_test]
                  val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
                  test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1).cuda()
                  tot = torch.zeros(1)
                  tot = tot.cuda()
                  accs = []
                  patience = 20
                  print('-' * 100)
                  cnt_wait = 0
                  for i in range(10):
                        idx_train = torch.load("./data/fewshot_cora/{}-shot_cora/{}/idx.pt".format(self.shot_num,i)).type(torch.long).cuda()
                        print('idx_train',idx_train)
                        train_lbls = torch.load("./data/fewshot_cora/{}-shot_cora/{}/labels.pt".format(self.shot_num,i)).type(torch.long).squeeze().cuda()
                        print("true",i,train_lbls)
                        best = 1e9
                        pat_steps = 0
                        best_acc = torch.zeros(1)
                        best_acc = best_acc.cuda()
                        pretrain_embs = embeds[0, idx_train]
                        for _ in range(50):
                              self.DownPrompt.train()
                              self.optimizer.zero_grad()
                              prompt_feature = self.feature_prompt(self.features)
                              # prompt_feature = self.feature_prompt(self.data.x)
                              # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
                              embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
                              pretrain_embs1 = embeds1[0, idx_train]
                              logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().cuda()
                              loss = self.criterion(logits, train_lbls)
                              if loss < best:
                                    best = loss
                                    cnt_wait = 0
                              else:
                                    cnt_wait += 1
                                    if cnt_wait == patience:
                                          print('Early stopping at '+str(_) +' eopch!')
                                          break
                              
                              loss.backward(retain_graph=True)
                              self.optimizer.step()

                        prompt_feature = self.feature_prompt(self.features)
                        embeds1, _ = self.Preprompt.embed(prompt_feature, self.sp_adj, True, None, False)
                        test_embs1 = embeds1[0, self.idx_test]
                        print('idx_test', self.idx_test)
                        logits = self.DownPrompt(test_embs, test_embs1, train_lbls)
                        preds = torch.argmax(logits, dim=1)
                        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                        accs.append(acc * 100)
                        print('acc:[{:.4f}]'.format(acc))
                        tot += acc

                  print('-' * 100)
                  print('Average accuracy:[{:.4f}]'.format(tot.item() / 10))
                  accs = torch.stack(accs)
                  print('Mean:[{:.4f}]'.format(accs.mean().item()))
                  print('Std :[{:.4f}]'.format(accs.std().item()))
                  print('-' * 100)
                  
            
            print("Node Task completed")


