from torch import nn, optim

import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ProG.utils import seed, seed_everything
from random import shuffle
# import pandas as pd
from ProG.prompt import Pipeline
from ProG.meta import MAML
from data_preprocess import load_tasks
from torch_geometric.loader import DataLoader
import torchmetrics

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

        support_loader = DataLoader(support.to_data_list(), batch_size=10, shuffle=True)
        query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=True)

        for _ in range(adapt_steps_meta_test):
            running_loss = 0.
            for batch_id, support_batch in enumerate(support_loader):
                support_preds = test_model(support_batch)
                support_loss = lossfn(support_preds, support_batch.y)
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()
                running_loss += support_loss.item()
                # print("{}/{} batch loss: {:.8f}".format(batch_id+1,len(support_loader),support_loss.item()))

                if batch_id == len(support_loader) - 1:
                    last_loss = running_loss / len(support_loader)  # loss per batch
                    print('{}/{} training loss: {:.8f}'.format(_, adapt_steps_meta_test, last_loss))
                    running_loss = 0.

        test_model.eval()
        metric = torchmetrics.classification.Accuracy(task="binary")  # , num_labels=2)
        for batch_id, query_batch in enumerate(query_loader):
            query_preds = test_model(query_batch)
            pre_class = torch.argmax(query_preds, dim=1)
            acc = metric(pre_class, query_batch.y)
            # print(f"Accuracy on batch {batch_id}: {acc}")

        acc = metric.compute()
        print("""\ttask pair ({}, {}) | Acc: {:.4} """.format(task_1, task_2, acc))
        metric.reset()


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
    gnn_type = ['TransformerConv']

    maml, opt, lossfn = model_components()
    epoch = 20  # 200
    adapt_steps = 20
    # meta_train_maml(epoch, maml, lossfn, opt, meta_train_task_id_list,
    #                 dataname, adapt_steps, K_shot=100)

    adapt_steps_meta_test = 2  # 00  # 50

    meta_test_adam(meta_test_task_id_list, dataname, 100, seed, maml,
                   adapt_steps_meta_test, lossfn)
