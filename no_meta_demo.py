from ProG.utils import seed_everything, seed

seed_everything(seed)

from ProG import PreTrain
from ProG.utils import mkdir, load_data4pretrain
from ProG.prompt import GNN, LightPrompt, HeavyPrompt
from torch import nn, optim
from ProG.data import multi_class_NIG
import torch
from ProG.eva import testing, testing_tune_answer
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
from ProG.eva import acc_f1_over_batches


# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(dataname, gnn_type, num_class, task_type='multi_class_classification', tune_answer=False):
    if task_type in ['multi_class_classification', 'regression']:
        input_dim, hid_dim = 100, 100
        lr, wd = 0.001, 0.00001
        tnpc = 100  # token number per class

        # load pre-trained GNN
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG = HeavyPrompt(token_dim=input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3)
        else:
            PG = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                         lr=lr,
                         weight_decay=wd)

        if task_type == 'regression':
            lossfn = nn.MSELoss(reduction='mean')
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            if task_type == 'regression':
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01,
                                    weight_decay=0.00001)
        else:
            answering, opi_answer = None, None

        return gnn, PG, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))


def pretrain():
    mkdir('./pre_trained_gnn/')

    pretext = 'GraphCL'  # 'GraphCL', 'SimGRACE'
    gnn_type = 'TransformerConv'  # 'GAT', 'GCN'
    dataname, num_parts, batch_size = 'CiteSeer', 200, 10

    print("load data...")
    graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    print("create PreTrain instance...")
    pt = PreTrain(pretext, gnn_type, input_dim, hid_dim, gln=2)

    print("pre-training...")
    pt.train(dataname, graph_list, batch_size=batch_size,
             aug1='dropN', aug2="permE", aug_ratio=None,
             lr=0.01, decay=0.0001, epochs=100)


def prompt_w_o_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification'):
    train, test, _, _ = multi_class_NIG(dataname, num_class)

    gnn, PG, opi, lossfn, _, _ = model_create(dataname, gnn_type, num_class, task_type)
    prompt_epoch = 200  # 200
    # training stage
    PG.train()
    emb0 = gnn(train.x, train.edge_index, train.batch)
    for j in range(prompt_epoch):
        pg_batch = PG.inner_structure_update()
        pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)

        # cross link between prompt and input graphs
        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        if task_type == 'multi_class_classification':
            sim = torch.softmax(dot, dim=1)
        elif task_type == 'regression':
            sim = torch.sigmoid(dot)  # 0-1
        else:
            raise KeyError("task type error!")

        train_loss = lossfn(sim, train.y)

        print('{}/{} training loss: {:.8f}'.format(j, prompt_epoch, train_loss.item()))

        opi.zero_grad()
        train_loss.backward()
        opi.step()

        if j % 5 == 0:
            PG.eval()
            res = testing(test, PG, gnn, task_type=task_type)
            if task_type == 'regression':
                print("""MAE: {:.4} | MSE: {:.4} """.format(res["mae"], res["mse"]))

            else:
                print("""Acc: {:.4} | Macro F1: {:.4} | Micro F1: {:.4}""".format(res["acc"],
                                                                                  res["mac_f1"],
                                                                                  res["mic_f1"]))
            PG.train()


def train_one_outer_epoch(epoch, train_loader, opi, lossfn, gnn, PG, answering):
    for j in range(1, epoch + 1):
        running_loss = 0.
        # bar2=tqdm(enumerate(train_loader))
        for batch_id, train_batch in enumerate(train_loader):  # bar2
            # print(train_batch)
            prompted_graph = PG(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # print(graph_emb)
            pre = answering(graph_emb)
            # print(pre)
            train_loss = lossfn(pre, train_batch.y)
            # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #                                                                     train_loss.item()))

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            if batch_id % 5 == 4:  # report every 5 updates
                last_loss = running_loss / 5  # loss per batch
                # bar2.set_description('answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
                #                                                                     last_loss))
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, epoch, batch_id, len(train_loader), last_loss))

                running_loss = 0.


def prompt_w_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification'):
    _, test_batch, train_list, test_list = multi_class_NIG(dataname, num_class, shots=100)

    train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=10, shuffle=True)

    gnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(dataname, gnn_type, num_class, task_type, True)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = 10
    answer_epoch = 1  # 50
    prompt_epoch = 1  # 50

    # training stage
    for i in range(1, outer_epoch + 1):
        print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG, answering)

        print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG, answering)

        # testing stage
        answering.eval()
        PG.eval()
        acc_f1_over_batches(test_loader, PG, gnn, answering, num_class, task_type)


if __name__ == '__main__':
    # pretrain()
    # prompt_w_o_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification')
    prompt_w_h(dataname="CiteSeer", gnn_type="TransformerConv", num_class=6, task_type='multi_class_classification')
