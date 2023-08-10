
<p align="center">
  <img height="150" src="https://github.com/sheldonresearch/ProG/blob/main/Logo.jpg?sanitize=true" />
</p>

---

![](https://img.shields.io/badge/Latest_version-v0.1.1-red)
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/PyTorch-v1.13.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.9-red)


| **[Website](https://graphprompt.github.io/)** | **[Paper](https://arxiv.org/abs/2307.01504)** | **[Video](https://www.youtube.com/watch?v=MFL0ynk1BKs)** | **[Raw Code](https://anonymous.4open.science/r/mpg/README.md)** |


**ProG** (_Prompt Graph_) is a library built upon PyTorch to easily conduct single or multiple task prompting for a
pre-trained Graph Neural Networks (GNNs). The idea is derived from the paper: Xiangguo Sun, Hong Cheng, JIa Li,
etc. [All in One: Multi-task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504). KDD2023, in which
they released the raw
codes at [Click](https://anonymous.4open.science/r/mpg/README.md). This repository is a **polished version** of the raw
codes
with **[Extremely Huge Changes and Updates:](https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023)**

- [Historical Update Logs](https://github.com/sheldonresearch/ProG/blob/main/History.md)
- [Historical Releases](https://github.com/sheldonresearch/ProG/releases)
- [Differences](https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023)



## Quick Start


### Package Dependencies

- PyTorch 1.13.1
- torchmetrics 0.11.4
- torch_geometric 2.2.0

### Pre-train your GNN model

The following codes present a simple example on how to pre-train a GNN model via GraphCL. You can also find a integrated function ``pretrain()`` in ``no_meta_demo.py``.
```python
from ProG.utils import mkdir, load_data4pretrain
from ProG import PreTrain

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


```

### Create Relative Models

```python
from ProG.prompt import GNN, LightPrompt
from torch import nn, optim
import torch

# load pre-trained GNN
gnn = GNN(100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type="TransformerConv")
pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format("CiteSeer", "TransformerConv")
gnn.load_state_dict(torch.load(pre_train_path))
print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
for p in gnn.parameters():
    p.requires_grad = False

# prompt with hand-crafted answering template (no answering head tuning)
PG = LightPrompt(token_dim=100, token_num_per_group=100, group_num=6, inner_prune=0.01)

opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                 lr=0.001, weight_decay=0.00001)

lossfn = nn.CrossEntropyLoss(reduction='mean')

```
The above codes are also integrated as a function ``model_create(dataname, gnn_type, num_class, task_type)`` in this project. 

### Prompt learning with hand-crafted answering template
```python
from ProG.data import multi_class_NIG
import torch

train, test,_,_ = multi_class_NIG(dataname, num_class)
gnn, PG, opi, lossfn, _, _ = model_create(dataname, gnn_type, num_class, task_type)
prompt_epoch = 200  # 200
# training stage
PG.train()
emb0 = gnn(train.x, train.edge_index, train.batch)
for j in range(prompt_epoch):
    pg_batch = PG.inner_structure_update()
    pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
    dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
    sim = torch.softmax(dot, dim=1)
    train_loss = lossfn(sim, train.y)
    print('{}/{} training loss: {:.8f}'.format(j, prompt_epoch, train_loss.item()))
    opi.zero_grad()
    train_loss.backward()
    opi.step()
```
### More Detailed Tutorial
For more detailed usage examples w.r.t ``prompt with answer tuning``, ``prompt with meta-learning`` etc. Please check the demo in:

- ``no_meta_demo.py``
- ``meta_demo.py``

### Compare this new implementation with the raw code

```
Multi-class node classification (100-shots)

                      |      CiteSeer     |
                      |  ACC  | Macro-F1  |
==========================================|
reported in the paper | 80.50 |   80.05   |
(Prompt)              |                   |
------------------------------------------|
this version code     | 81.00 |   81.23   |
(Prompt)              |   (run one time)  | 
==========================================|
reported in the paper | 80.00 ｜  80.05   ｜
(Prompt w/o h)        |                   ｜
------------------------------------------|
this version code     | 79.78 ｜  80.01   ｜
(Prompt w/o h)        |   (run one time)  ｜
==========================================|

```
**Note:**
- Kindly note that the comparison takes the same pre-trained pth. The final results may vary depending on different
pre-training states 
- The above table is copied from this blog: https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023


## Citation 
bibtex
```
@inproceedings{sun2023all,
  title={All in One: Multi-Task Prompting for Graph Neural Networks},
  author={Sun, Xiangguo and Cheng, Hong and Li, Jia and Liu, Bo and Guan, Jihong},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'23)},
  year={2023}
}

```


## Contact

- For More Information, Further discussion, Contact: [Website](https://graphprompt.github.io/)
- Email: xiangguosun at cuhk dot edu dot hk