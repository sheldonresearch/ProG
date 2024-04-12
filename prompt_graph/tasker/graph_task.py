import torch
from prompt_graph.data import load4graph,load4node
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva
import time

class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.load_data()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC']:
            self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset, _= load4graph(self.dataset_name, self.shot_num)

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

    def GpromptTrain(self, train_loader, center):
        self.prompt.train()
        total_loss = 0.0

        for batch in train_loader:
            
            # archived code for complete prototype embeddings of each labels. Not as well as batch version
            # # compute the prototype embeddings of each type of label

            # for index, batch in enumerate(train_loader):     
            #     self.pg_opi.zero_grad() 
            #     batch = batch.to(self.device)
                
            #     out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')

            #     if(index == 0):
            #         total_embed_of_each_label = torch.zeros(self.output_dim, out.size(1)).to(self.device)
            #         totoal_num_of_each_label = torch.zeros(self.output_dim, 1).to(self.device)           

            #     # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            #     b_total_embed, b_total_num = batch_total_embedding(out, batch.y, self.output_dim)
            #     total_embed_of_each_label+=b_total_embed
            #     totoal_num_of_each_label+=b_total_num
            
            # # center = total_embed_of_each_label 
            # for i in range(self.output_dim): # self.output_dim = label number
            #     if(totoal_num_of_each_label[i].item()==0):
            #         continue
            #     else:
            #         center[i] /= totoal_num_of_each_label[i] # compute average embs of each type of label, where the sample num is not 0.

            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center=center_embedding(out,batch.y, self.output_dim)
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)  
            loss.backward()  
            self.pg_opi.step()  
            total_loss += loss.item()

        return total_loss / len(train_loader), center


    def run(self):

        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        print("prepare data is finished!")
        best_val_acc = final_test_acc = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            if self.prompt_type == 'None':
                loss = self.Train(train_loader)
                test_acc = GNNGraphEva(test_loader, self.gnn, self.answering, self.device)
                val_acc = GNNGraphEva(val_loader, self.gnn, self.answering, self.device)
            elif self.prompt_type == 'All-in-one':
                loss = self.AllInOneTrain(train_loader)
                test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                val_acc, F1 = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            elif self.prompt_type in ['GPF', 'GPF-plus']:
                loss = self.GPFTrain(train_loader)
                test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
                val_acc = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.device)
            elif self.prompt_type =='Gprompt':
                loss, center = self.GpromptTrain(train_loader)
                test_acc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.device)
                val_acc = GpromptEva(val_loader, self.gnn, self.prompt, center, self.device)
                    

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print("Epoch {:03d}/{:03d}  |  Time(s) {:.4f}| Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, self.epochs, time.time() - t0, loss, val_acc, test_acc))
            
        print(f'Final Test: {final_test_acc:.4f}')
        
        print("Graph Task completed")

        

        
