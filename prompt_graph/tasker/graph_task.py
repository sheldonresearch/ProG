import torch
from prompt_graph.data import load4graph, load4node, graph_sample_and_save
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva
import time
import os 
import numpy as np

class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.load_data()
        # self.create_few_data_folder()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()

    def create_few_data_folder(self):
            # 创建文件夹并保存数据
            for k in range(1, 11):
                  k_shot_folder = './Experiment/sample_data/Graph/'+ self.dataset_name +'/' + str(k) +'_shot'
                  os.makedirs(k_shot_folder, exist_ok=True)
                  
                  for i in range(1, 6):
                        folder = os.path.join(k_shot_folder, str(i))
                        os.makedirs(folder, exist_ok=True)
                        graph_sample_and_save(self.dataset, k, folder, self.output_dim)
                        print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR']:
            self.input_dim, self.output_dim, self.dataset= load4graph(self.dataset_name, self.shot_num)

    def Train(self, train_loader):
        self.gnn.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
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
        # answer_epoch = 5  # 50
        # prompt_epoch = 1  # 50        
        
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

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None

        for batch in train_loader:
            
            # archived code for complete prototype embeddings of each labels. Not as well as batch version
            # # compute the prototype embeddings of each type of label

            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out,batch.y, self.output_dim)
            # 累积中心向量和样本数
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)  
            loss.backward()  
            self.pg_opi.step()  
            total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers


    def run(self):
        test_accs = []
        for i in range(1, 6):
           
            idx_train = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
            print('idx_train',idx_train)
            train_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            print("true",i,train_lbls)

            idx_test = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
            test_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
        
            train_dataset = self.dataset[idx_train]
            test_dataset = self.dataset[idx_test]
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            print("prepare data is finished!")
  
            patience = 20
            best = 1e9
            cnt_wait = 0
           


            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                if self.prompt_type == 'None':
                    loss = self.Train(train_loader)
                elif self.prompt_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.prompt_type =='Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                           
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
                test_acc = GNNGraphEva(test_loader, self.gnn, self.answering, self.device)
            elif self.prompt_type == 'All-in-one':
                test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            elif self.prompt_type in ['GPF', 'GPF-plus']:
                test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
            elif self.prompt_type =='Gprompt':
                test_acc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.device)

            print("test accuracy {:.4f} ".format(test_acc))                        
            test_accs.append(test_acc)
        
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)    
        print(" Final best | test Accuracy {:.4f} | std {:.4f} ".format(mean_test_acc, std_test_acc))         
        
        print("Graph Task completed")

        

        
