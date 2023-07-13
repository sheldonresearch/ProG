from itertools import product
from collections import defaultdict

"""
Cora
label 0 total num subset 351
label 1 total num subset 217
label 2 total num subset 418
label 3 total num subset 818
label 4 total num subset 426
label 5 total num subset 298
label 6 total num subset 180

use task 2,3,4 for node | 9,10,11 for edge | 16,17,18 for graph tasks



CiteSeer:
use last two


computers:
label 5,9 insufficient
7,8
17,18


"""


class ParReddit2(object):
    def __init__(self):
        self.dataname = 'Reddit2'
        self.adapt_lr = None
        self.meta_lr = None
        self.adapt_steps = None
        self.epoch = None
        self.adapt_steps_meta_test = None
        self.K_shot = None

        self.exp_type = defaultdict(dict)
        self.para_set = None
        self.set_parameters()

    def set_parameters(self):
        self.adapt_lr, self.meta_lr, self.adapt_steps, self.epoch, self.adapt_steps_meta_test, self.K_shot, self.exp_type = self.macro_pars()
        self.para_set = self.micro_pars()

    def macro_pars(self):
        adapt_lr = 0.01  # only support meta-training, for meta-test we have builtin lr in meta_test_adam() (0.01)
        meta_lr = 0.001  # only support meta-training, for meta-test we have builtin lr in meta_test_adam() (0.01)
        adapt_steps = 2  # adapt step within meta-training (inner loop)
        epoch = 50  # meta-training epoch (outter loop)
        adapt_steps_meta_test = 40  # meta-test epoch
        K_shot = 100

        exp_type = defaultdict(dict)
        """
        node-level: [0,41)
        edge-level: [41,82)
        graph-level: [82,123)
        """

        exp_type['graph_level'] = {
            'meta_train_tasks': [82, 83, 84, 85],  # [i for i in range(82, 121)], OOM for so many tasks
            'meta_test_tasks': {
                'graph2graph': [121, 122],
                'graph2node': [39, 40]
            }
        }

        exp_type['edge_level'] = {
            'meta_train_tasks': [41, 42, 43, 44],  # [i for i in range(41, 80)],, OOM for so many tasks
            'meta_test_tasks': {
                'edge2edge': [80, 81],
                'edge2node': [39, 40]
            }
        }

        exp_type['node_level'] = {
            'meta_train_tasks': [0, 1, 2, 3],  # [i for i in range(0, 39)],, OOM for so many tasks
            'meta_test_tasks': {
                'node2node': [39, 40]
            }
        }

        return adapt_lr, meta_lr, adapt_steps, epoch, adapt_steps_meta_test, K_shot, exp_type

    def micro_pars(self):
        para_set = set()
        pre_train_method = ['None', 'GraphCL', 'SimGRACE']
        with_prompt = [False]
        meta_learning = [False]
        gnn_type = ['GAT', 'GCN', 'TransformerConv']
        """
        first round: 
            with_prompt=False,meta_learning=False
            save model state dict for various gnns and various pre-train (largely project head)
            save related evaluation results
        second round
            with_prompt=True,meta_learning=False
            load the above model
            prompt-tuning
            save prompt model and save related evaluation results
        third round
            meta_learning=True to see whether has any boost.
        
        
        
        current: K=100 (200 in total)
        largest=700
        TransformerConv: OOM
        GCN: work well
        GAT: 
        
        
        """
        pre_epoch = None

        for para_list in product(pre_train_method, with_prompt, meta_learning, gnn_type):
            pre_train_method, with_prompt, meta_learning, gnn_type = para_list
            if pre_train_method == 'None':
                with_prompt, meta_learning = False, False
                pre_train_path = None
            else:
                if pre_train_method == 'GraphCL' and gnn_type == 'GAT':
                    pre_epoch = 100
                elif pre_train_method == 'GraphCL' and gnn_type == 'GCN':
                    pre_epoch = 80
                elif pre_train_method == 'GraphCL' and gnn_type == 'TransformerConv':
                    pre_epoch = 90
                elif pre_train_method == 'SimGRACE' and gnn_type == 'GAT':
                    pre_epoch = 70
                elif pre_train_method == 'SimGRACE' and gnn_type == 'GCN':
                    pre_epoch = 100
                elif pre_train_method == 'SimGRACE' and gnn_type == 'TransformerConv':
                    pre_epoch = 90

                pre_train_path = "./pre_trained_gnn/{}.{}.{}.dataupdate_{}.epoch_{}.pth".format(self.dataname,
                                                                                                pre_train_method,
                                                                                                gnn_type, 1, pre_epoch)

            para_set.add((pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path))

        return sorted(para_set, reverse=True)


class ParPlanetoid(object):
    def __init__(self, dataname: str, round=1):
        # dataname='CiteSeer','PubMed', 'Cora'
        # macro_pars

        if dataname not in ['CiteSeer', 'PubMed', 'Cora', 'Computers']:
            raise KeyError("dataname is wrong!")

        self.round = round
        self.dataname = dataname
        self.adapt_lr = None
        self.meta_lr = None
        self.adapt_steps = None
        self.epoch = None
        self.adapt_steps_meta_test = None
        self.K_shot = None
        self.meta_test_type_meta_test_task_id_list = None
        self.set_macro_pars()

        # micro_pars
        self.micro_pars = None
        self.set_micro_pars(round=1)

    def set_macro_pars(self):
        self.adapt_lr = 0.01  # only support meta-training, for meta-test we have builtin lr in meta_test_adam() (0.01)
        self.meta_lr = 0.001  # only support meta-training, for meta-test we have builtin lr in meta_test_adam() (0.01)
        self.adapt_steps = 2  # adapt step within meta-training (inner loop)
        self.epoch = 200  # meta-training epoch (outter loop)
        if self.round == 1:
            self.adapt_steps_meta_test = 20  # meta-test epoch: when round=1 it is 20, when round=2, it is 50
        elif self.round == 2:
            self.adapt_steps_meta_test = 50  # meta-test epoch: when round=1 it is 20, when round=2, it is 50
        self.K_shot = 100

        if self.dataname == 'PubMed':
            self.meta_test_type_meta_test_task_id_list = list(zip(['node', 'edge', 'graph'],
                                                                  [[1, 2], [4, 5], [7, 8]]))
        elif self.dataname == 'CiteSeer':
            self.meta_test_type_meta_test_task_id_list = list(zip(['node', 'edge', 'graph'],
                                                                  # [[4, 5], [10, 11], [16, 17]]))
                                                                  [[4, 5], [6, 7], [14, 15]]))

        elif self.dataname == 'Cora':
            self.meta_test_type_meta_test_task_id_list = list(zip(['node', 'edge', 'graph'],
                                                                  [[3, 4], [9, 10], [16, 17]]))

        elif self.dataname == 'Computers':
            self.meta_test_type_meta_test_task_id_list = list(zip(['node', 'edge', 'graph'],
                                                                  [[7, 8], [17, 18], [27, 28]]))

        else:
            raise KeyError("dataname is wrong!")

    def set_micro_pars(self, round=1):

        micro_pars = set()
        if round == 1:
            pre_train_method = ['None', 'GraphCL', 'SimGRACE']
            with_prompt = [False]
            meta_learning = [False]
            gnn_type = ['GAT', 'GCN', 'TransformerConv']

        elif round == 2:
            pre_train_method = ['GraphCL', 'SimGRACE']
            with_prompt = [True]
            meta_learning = [False]
            gnn_type = ['GAT', 'GCN', 'TransformerConv']
        elif round == 3:
            raise NotImplemented('config.ParPlanetoid.set_micro_pars not implemented for round=3!')
        else:
            raise KeyError('round {} is not acceptable'.format(round))

        """
        first round: 
            with_prompt=False,meta_learning=False
            save model state dict for various gnns and various pre-train (largely project head)
            save related evaluation results
        second round
            with_prompt=True,meta_learning=False
            load the above model
            prompt-tuning
            save prompt model and save related evaluation results
        third round
            meta_learning=True to see whether has any boost.



        current: K=100 (200 in total)
        largest=700
        TransformerConv: OOM
        GCN: work well
        GAT: 


        """

        for para_list in product(pre_train_method, with_prompt, meta_learning, gnn_type):
            pre_train_method, with_prompt, meta_learning, gnn_type = para_list
            if pre_train_method == 'None':
                with_prompt, meta_learning = False, False
                pre_train_path = None
            else:
                pre_train_path = "./pre_trained_gnn/{}.{}.{}.pth".format(self.dataname,
                                                                         pre_train_method,
                                                                         gnn_type)

            micro_pars.add((pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path))

        self.micro_pars = sorted(micro_pars, reverse=True)


if __name__ == '__main__':
    from data_load_Planetoid import induced_graph_2_K_shot
    import pickle as pk

    dataname = 'CiteSeer'
    task_1, task_2 = 14, 15
    meta_stage = 'test'
    K_shot = 100
    task_1_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_1, meta_stage)
    task_1_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)
    task_2_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_2, meta_stage)
    task_2_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

    with (open(task_1_support, 'br') as t1s,
          open(task_1_query, 'br') as t1q,
          open(task_2_support, 'br') as t2s,
          open(task_2_query, 'br') as t2q):
        t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
        support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=0)

        t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
        query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=0)

        print(support)
        print(query)

    #
    # args = ParPlanetoid(dataname)
    #
    #
    # round = 1  # 2,3
    #
    # args.set_micro_pars(round=round)
    # print(list(args.meta_test_type_meta_test_task_id_list))
    #
    # for paras in args.micro_pars:
    #     pre_train_method, with_prompt, meta_learning, gnn_type, pre_train_path = paras
    #
    #     for meta_test_type, meta_test_task_id_list in args.meta_test_type_meta_test_task_id_list:
    #         print(pre_train_method,meta_test_type,meta_test_task_id_list)
