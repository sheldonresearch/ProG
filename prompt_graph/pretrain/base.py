import os
import torch
from prompt_graph.model import build_gnn
from torch.optim import Adam

class PreTrain(torch.nn.Module):
    def __init__(self, gnn_type='TransformerConv', dataset_name = 'Cora', input_dim=128, hid_dim = 128, gln = 2, num_epoch = 1000, device : int = 5, graph_list=None, num_workers=0):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:' + str(device))
        elif os.environ.get('PROG_USE_MPS') == '1' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.graph_list = graph_list
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim =hid_dim
        self.learning_rate = 0.001
        self.weight_decay = 0.00005
        self.num_workers = num_workers
    
    def initialize_gnn(self, input_dim, hid_dim):
        self.gnn = build_gnn(self.gnn_type, input_dim, hid_dim, self.num_layer)
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


        
#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes
