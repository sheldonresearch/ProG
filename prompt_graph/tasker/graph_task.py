import torch
from prompt_graph.data import load4graph, load4node, graph_sample_and_save
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss,constraint
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva, GPPTGraphEva
import time
import os 
import numpy as np

class GraphTask(BaseTask):
    def __init__(input_dim, output_dim, dataset, self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.task_type = 'GraphTask'
        # self.load_data()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        self.create_few_data_folder()
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
                        if not os.path.exists(folder):
                            os.makedirs(folder, exist_ok=True)
                            graph_sample_and_save(self.dataset, k, folder, self.output_dim)
                            print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'ogbg-ppa','DD']:
            self.input_dim, self.output_dim, self.dataset= load4graph(self.dataset_name, self.shot_num)

    def node_degree_as_features(self, data_list):
        from torch_geometric.utils import degree
        for data in data_list:
            # 计算所有节点的度数，这将返回一个张量
            deg = degree(data.edge_index[0], dtype=torch.long)

            # 将度数张量变形为[nodes, 1]以便与其他特征拼接
            deg = deg.view(-1, 1).float()
            
            # 如果原始数据没有节点特征，可以直接使用度数作为特征
            if data.x is None:
                data.x = deg
            else:
                # 将度数特征拼接到现有的节点特征上
                data.x = torch.cat([data.x, deg], dim=1)

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
        
    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        #we update answering and prompt alternately.
        
        # answer_epoch = 1  # 50
        # prompt_epoch = 1  # 50
        # answer_epoch = 5  # 50  #PROTEINS # COX2
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

    def GPPTtrain(self, train_loader):
        self.prompt.train()
        for batch in train_loader:
            temp_loss=torch.tensor(0.0,requires_grad=True).to(self.device)
            graph_list = batch.to_data_list()
            for index, graph in enumerate(graph_list):
                graph=graph.to(self.device)              
                node_embedding = self.gnn(graph.x,graph.edge_index)
                out = self.prompt(node_embedding, graph.edge_index)
                loss = self.criterion(out, torch.full((1,graph.x.shape[0]), graph.y.item()).reshape(-1).to(self.device))
                temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            temp_loss = temp_loss/(index+1)
            self.pg_opi.zero_grad()
            temp_loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
        return temp_loss.item()

    def run(self):
        test_accs = []
        batch_best_loss = []
        for i in range(1, 6):
            idx_train = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
            print('idx_train',idx_train)
            train_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            print("true",i,train_lbls)

            idx_test = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
            test_lbls = torch.load("./Experiment/sample_data/Graph/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
        
            train_dataset = self.dataset[idx_train]
            test_dataset = self.dataset[idx_test]

            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                from torch_geometric.data import Batch
                train_dataset = [train_g for train_g in train_dataset]
                test_dataset = [test_g for test_g in test_dataset]
                processed_dataset = [g for g in self.dataset]
                self.node_degree_as_features(train_dataset)
                self.node_degree_as_features(test_dataset)
                self.node_degree_as_features(processed_dataset)
                processed_dataset = Batch.from_data_list([g for g in processed_dataset])
                self.input_dim = train_dataset[0].x.size(1)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            print("prepare data is finished!")
  
            patience = 20
            best = 1e9
            cnt_wait = 0
           
            if self.prompt_type == 'All-in-one':
                # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
                # self.prompt_epoch = 1
                self.answer_epoch = 5
                self.prompt_epoch = 1
                self.epochs = int(self.epochs/self.answer_epoch)
            elif self.prompt_type == 'GPPT':
                # initialize the GPPT hyperparametes via graph data
                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                    # total_num_nodes = sum([data.num_nodes for data in train_dataset])
                    # train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                    # self.gppt_loader = DataLoader(processed_dataset, batch_size=1, shuffle=True)
                    # for i, batch in enumerate(self.gppt_loader):
                    #     if(i==0):
                    #         node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                    #     else:                   
                    #         node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                    
                    # node_embedding = self.gnn(processed_dataset.x.to(self.device), processed_dataset.edge_index.to(self.device))
                    # node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)             
                    # self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                    total_num_nodes = sum([data.num_nodes for data in train_dataset])
                    train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                    self.gppt_loader = DataLoader(processed_dataset.to_data_list(), batch_size=1, shuffle=False)
                    for i, batch in enumerate(self.gppt_loader):
                        if(i==0):
                            node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                            node_embedding = self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))
                        else:                   
                            node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                            node_embedding = torch.concat([node_embedding,self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))],dim=0)
                    
                    node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                    self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)                    
                else:
                    train_node_ids = torch.arange(0,train_dataset.x.shape[0]).squeeze().to(self.device)
                    self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
                    for i, batch in enumerate(self.gppt_loader):
                        if(i==0):
                            node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                        else:                   
                            node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                    
                    node_embedding = self.gnn(self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device))
                    node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)             
                    self.prompt.weigth_init(node_embedding,self.dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                # from torch_geometric.nn import global_mean_pool
                # self.gppt_pool = global_mean_pool
                # train_ids = torch.nonzero(idx_train, as_tuple=False).squeeze()
                # self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)          
                # for i, batch in enumerate(self.gppt_loader):
                #     batch.to(self.device)
                #     node_embedding = self.gnn(batch.x, batch.edge_index)
                #     if(i==0):
                #         graph_embedding = self.gppt_pool(node_embedding,batch.batch.long())
                #     else:
                #         graph_embedding = torch.concat([graph_embedding,self.gppt_pool(node_embedding,batch.batch.long())],dim=0)
                

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                if self.prompt_type == 'None':
                    loss = self.Train(train_loader)
                elif self.prompt_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.prompt_type =='Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                elif self.prompt_type =='GPPT':
                    loss = self.GPPTtrain(train_loader)
                           
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
            batch_best_loss.append(loss)
            
            if self.prompt_type == 'None':
                test_acc = GNNGraphEva(test_loader, self.gnn, self.answering, self.device)
            elif self.prompt_type == 'All-in-one':
                test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            elif self.prompt_type in ['GPF', 'GPF-plus']:
                test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
            elif self.prompt_type =='Gprompt':
                test_acc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.device)
            elif self.prompt_type =='GPPT':
                test_acc = GPPTGraphEva(test_loader, self.gnn, self.prompt, self.device)

            print("test accuracy {:.4f} ".format(test_acc)) 
            print("best_loss",  batch_best_loss)                        
            test_accs.append(test_acc)
        
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)    
        print(" Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))   

        print(self.pre_train_type, self.gnn_type, self.prompt_type, " Graph Task completed")
        mean_best = np.mean(batch_best_loss)

        return  mean_best, mean_test_acc, std_test_acc

        

        
