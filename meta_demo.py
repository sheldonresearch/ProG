from torch import nn, optim

import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ProG.utils import seed, seed_everything
from random import shuffle
# import pandas as pd
from config import ParPlanetoid
from ProG.prompt import Pipeline
from ProG.meta import MAML
from data_preprocess import load_tasks

seed_everything(seed)


def meta_test_adam(meta_test_task_id_list,
                   dataname,
                   K_shot,
                   seed,
                   maml,
                   adapt_steps_meta_test,
                   lossfn):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, _ in load_tasks('test', task_pairs, dataname, K_shot, seed):

        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.00001)

        test_model.train()

        for _ in range(adapt_steps_meta_test):
            # TODO: need to support batch compute
            support_preds = test_model(support)
            support_loss = lossfn(support_preds, support.y)
            if _ % 5 == 0:
                print('{}/{} training loss: {:.8f}'.format(_,
                                                           adapt_steps_meta_test,
                                                           support_loss.item()))
            test_opi.zero_grad()
            support_loss.backward()
            test_opi.step()

        test_model.eval()
        # TODO: need to support batch compute
        query_preds = test_model(query)

        pre_class = torch.argmax(query_preds, dim=1)
        acc = accuracy_score(query.y, pre_class)
        print("""\ttask pair ({}, {}) | Acc: {:.4} """.format(task_1, task_2, acc))
        # f1 = f1_score(query.y, pre_class, average='binary')
        # auc = roc_auc_score(query.y, query_preds[:, 1].detach().numpy())
        # print("""\ttask pair ({}, {}) | Acc: {:.4} | F1: {:.4} | ACU: {:.4}""".format(task_1, task_2, acc, f1, auc))
        # task_results.append([task_1, task_2, acc, f1, auc])
        #
        # if save_project_head:
        #     torch.save(test_model.project_head.state_dict(),
        #                "./project_head/{}.{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type, meta_test_type))
        #     print("project head saved! @./project_head/{}.{}.{}.{}.pth".format(dataname, pre_train_method, gnn_type,
        #                                                                        meta_test_type))

    # return task_results


def meta_train_maml(epoch, maml, lossfn, opt, meta_train_task_id_list, dataname, adapt_steps, K_shot=100):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)

    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]

    # meta-training
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0

        for task_1, task_2, support, query, total_num in load_tasks('train', task_pairs, dataname, K_shot, seed):
            pair_count = pair_count + 1

            learner = maml.clone()

            for _ in range(adapt_steps):  # adaptation_steps
                # TODO: need to support batch compute
                support_preds = learner(support)
                support_loss = lossfn(support_preds, support.y)
                learner.adapt(support_loss)

            query_preds = learner(query)
            query_loss = lossfn(query_preds, query.y)
            meta_train_loss += query_loss

        print('\tmeta_train_loss at epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / len(meta_train_task_id_list)
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


def model_components():
    """
    input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'

    :param args:
    :param round:
    :param pre_train_path:
    :param gnn_type:
    :param project_head_path:
    :return:
    """
    adapt_lr = 0.01
    meta_lr = 0.001
    model = Pipeline(input_dim=100, dataname='CiteSeer', gcn_layer_num=2, hid_dim=100, num_classes=2,  # 0 or 1
                     task_type="multi_label_classification",
                     token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv')

    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)
    lossfn = nn.CrossEntropyLoss(reduction='mean')

    return maml, opt, lossfn


if __name__ == '__main__':
    dataname = 'CiteSeer'
    # node-level: 0 1 2 3 4 5
    # edge-level: 6 7 8 9 10 11
    # graph-level: 12 13 14 15 16 17
    meta_train_task_id_list = [0, 1, 2, 3]
    meta_test_task_id_list = [4, 5]

    pre_train_method = 'GraphCL'
    with_prompt = [True]
    meta_learning = [True]
    gnn_type = ['TransformerConv']
    pre_train_path = ""

    maml, opt, lossfn = model_components()
    epoch = 20  # 200
    adapt_steps = 20
    meta_train_maml(epoch, maml, lossfn, opt, meta_train_task_id_list,
                    dataname, adapt_steps, K_shot=100)

    adapt_steps_meta_test = 200  # 50

    meta_test_adam(meta_test_task_id_list, dataname, 100, seed, maml,
                   adapt_steps_meta_test, lossfn)
