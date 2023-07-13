

## 13 JUL 2023

### Big Update!

Compared with the raw project released in the paper ([code](https://anonymous.4open.science/r/mpg/README.md)), 0.1.1 has **EXTREMELY HUGE CHANGES**, including but not limited to:

- totally rewrite the whole project, the changed code takes up >80% of the original version.
- extremely simplify the code
- totally changed the project structures, class names, and new function designs.
- adopt [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) for automatic accumulation over batches in the evaluation stage (e.g. Acc, F1 etc)
- (In Progress) gradually remove ``sklearn.metrics`` in the original version
- more clear prompt module. In the raw project, there are more than three different implementations for prompt, which
  are very messy. Here, we remove all these ugly codes and unify them with a LightPrompt and a HeavyPrompt.
- support batch training and testing in function: ``meta_test_adam``
- and More


An example to see how much have we reduced the complexity for the project:

**_Before MPG.Models.Pipeline_**

```python

class Pipeline(torch.nn.Module):
    def __init__(self, input_dim,
                 pre_train_path=None, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 frozen_gnn='all',
                 with_prompt=True, token_num=5, prune_thre=0.5, inner_prune=None, isolate_tokens=False,
                 frozen_project_head=False, pool_mode=1, gnn_type='GAT', project_head_path=None):

        super().__init__()
        self.with_prompt = with_prompt
        self.pool_mode = pool_mode
        if with_prompt:
            self.token_num = token_num
            self.prompt = Prompt(token_dim=input_dim, token_num=token_num, prune_thre=prune_thre,
                                 isolate_tokens=isolate_tokens, inner_prune=inner_prune)
        self.gnn = GCN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)

        self.project_head = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, num_classes),
            torch.nn.Softmax(dim=1))

        self.set_gnn_project_head(pre_train_path, frozen_gnn, frozen_project_head, project_head_path)

    def set_gnn_project_head(self, pre_train_path, frozen_gnn, frozen_project_head, project_head_path=None):
        if pre_train_path:
            self.gnn.load_state_dict(torch.load(pre_train_path))
            print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))

        if project_head_path:
            self.project_head.load_state_dict(torch.load(project_head_path))
            print("successfully load project_head! @ {}".format(project_head_path))

        if frozen_gnn == 'all':
            for p in self.gnn.parameters():
                p.requires_grad = False
        elif frozen_gnn == 'none':
            for p in self.gnn.parameters():
                p.requires_grad = True
        else:
            pass

        if frozen_project_head:
            for p in self.project_head.parameters():
                p.requires_grad = False

    def forward(self, graph_batch: Batch):
        num_graphs = graph_batch.num_graphs
        if self.with_prompt:
            xp, xp_edge_index, batch_one, batch_two = self.prompt(graph_batch)

            if self.pool_mode == 1:
                graph_emb = self.gnn(xp, xp_edge_index, batch_one)
                pre = self.project_head(graph_emb)
                return pre
            elif self.pool_mode == 2:
                emb = self.gnn(xp, xp_edge_index, batch_two)
                graph_emb = emb[0:num_graphs, :]
                prompt_emb = emb[num_graphs:, :]
                com_emb = graph_emb - prompt_emb
                pre = self.project_head(com_emb)
                return pre
        else:
            graph_emb = self.gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            pre = self.project_head(graph_emb)
            return pre

```

**_Now ProG.prompt.Pipeline_**

```Python
class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'):

        super().__init__()
        # load pre-trained GNN
        self.gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        self.gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in self.gnn.parameters():
            p.requires_grad = False

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch: Batch):
        prompted_graph = self.PG(graph_batch)
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre

```


Another case to present how hard we tried to simplify the codes:

**_Prompt Before_**

```Python
class MPG.Models.Prompt(torch.nn.Module):
    def __init__(self, token_dim, token_num, prune_thre=0.9, isolate_tokens=False, inner_prune=None):
        super(Prompt, self).__init__()
        self.prune_thre = prune_thre
        if inner_prune is None:
            self.inner_prune = prune_thre
        else:
            self.inner_prune = inner_prune
        self.isolate_tokens = isolate_tokens
        self.token_x = torch.nn.Parameter(torch.empty(token_num, token_dim))
        self.initial_prompt()

    def initial_prompt(self, init_mode='kaiming_uniform'):
        if init_mode == 'metis':  # metis_num = token_num
            self.initial_prompt_with_metis()
        elif init_mode == 'node_labels':  # label_num = token_num
            self.initial_prompt_with_node_labels()
        elif init_mode == 'orthogonal':
            torch.nn.init.orthogonal_(self.token_x, gain=torch.nn.init.calculate_gain('leaky_relu'))
        elif init_mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.token_x, gain=torch.nn.init.calculate_gain('tanh'))
        elif init_mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.token_x, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        elif init_mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.token_x, nonlinearity='leaky_relu')
        elif init_mode == 'uniform':
            torch.nn.init.uniform_(self.token_x, 0.99, 1.01)
        else:
            raise KeyError("init_mode {} is not defined!".format(init_mode))

    def initial_prompt_with_metis(self, data=None, save_dir=None):
        if data is None:
            raise KeyError("you are calling initial_prompt_with_metis with empty data")
        metis = ClusterData(data=data, num_parts=self.token_x.shape[0], save_dir=save_dir)
        b = Batch.from_data_list(list(metis))
        x = global_mean_pool(b.x, b.batch)
        self.token_x.data = x

    def initial_prompt_with_node_labels(self, data=None):
        x = global_mean_pool(data.x, batch=data.y)
        self.token_x.data = x

    def forward(self, graph_batch: Batch):
        num_tokens = self.token_x.shape[0]
        node_x = graph_batch.x
        num_nodes = node_x.shape[0]
        num_graphs = graph_batch.num_graphs
        node_batch = graph_batch.batch
        token_x_repeat = self.token_x.repeat(num_graphs, 1)
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)])
        batch_one = torch.cat([node_batch, token_batch], dim=0)
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)]) + num_graphs
        batch_two = torch.cat([node_batch, token_batch], dim=0)
        edge_index = graph_batch.edge_index

        token_dot = torch.mm(self.token_x, torch.transpose(node_x, 0, 1))  # (T,d) (d, N)--> (T,N)
        token_sim = torch.sigmoid(token_dot)  # 0-1
        cross_adj = torch.where(token_sim < self.prune_thre, 0, token_sim)
        tokenID2nodeID = cross_adj.nonzero().t().contiguous()
        batch_value = node_batch[tokenID2nodeID[1]]
        new_token_id_in_cross_edge = tokenID2nodeID[0] + num_nodes + num_tokens * batch_value
        tokenID2nodeID[0] = new_token_id_in_cross_edge
        cross_edge_index = tokenID2nodeID
        if self.isolate_tokens:
            new_edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
        else:
            token_dot = torch.mm(self.token_x, torch.transpose(self.token_x, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            tokenID2tokenID = inner_adj.nonzero().t().contiguous()
            inner_edge_index = torch.cat([tokenID2tokenID + num_nodes + num_tokens * i for i in range(num_graphs)],
                                         dim=1)
            new_edge_index = torch.cat([edge_index, cross_edge_index, inner_edge_index], dim=1)

        new_edge_index, _ = add_self_loops(new_edge_index)
        new_edge_index = to_undirected(new_edge_index)
        edge_index_xp = sort_edge_index(new_edge_index)

        return (torch.cat([node_x, token_x_repeat], dim=0),
                edge_index_xp,
                batch_one,
                batch_two)


# prompt_w_o_h.py 
class PromptGraph(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_class, class_num, inner_prune=None):
        """
        :param token_dim:
        :param token_num:
        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(PromptGraph, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_class, token_dim)) for i in range(class_num)])

        for token in self.token_list:
            torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def token_view(self, ):
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch

```



**_Now ProG.prompt.LightPrompt_**

```Python

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch


```


**_Explore this version to find more surprising things!_**



**Evaluated results from this version:**

```
Multi-class node classification (100-shots)

                      |      CiteSeer     |
                      |  ACC  | Macro-F1  |
==========================================|
reported in the paper | 80.50 |   80.05   |
(Prompt)              |                   |
------------------------------------------|
this version code     | 81.00 |   --      |
(Prompt)              |   (run one time)  | 
==========================================|
reported in the paper | 80.00 ｜  80.05   ｜
(Prompt w/o h)        |                   ｜
------------------------------------------|
this version code     | 79.78 ｜  80.01   ｜
(Prompt w/o h)        |   (run one time)  ｜
==========================================|
--: hasn't implemented batch F1 in this version
```



### Future TODO List
- remove our self-implemented MAML module, replace it with a third-party meta library such as learn2learn or Torchmeta
- support sparse training
- support GPU
- support True Batch computing
- support GIN and more GNNs
- support more Pre-train methods such as GraphGPT
- test on large-scale
- support distributed computing
- support more tasks and data sets

**Full Changelog**: https://github.com/sheldonresearch/ProG/commits/latest