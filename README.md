<h1 align="center">
  <img height="150" src="/Logo.jpg?sanitize=true" />
</h1>





<p align="left">

![](https://img.shields.io/badge/Latest_version-v0.1.5-red)
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/PyTorch-v1.13.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.9-red)

</p>


<p align="left">
  
| **[Quick Start](#quick-start)** 
| **[Website](https://graphprompt.github.io/)** | **[Paper](https://arxiv.org/abs/2307.01504)**
| **[Video](https://www.youtube.com/watch?v=MFL0ynk1BKs)**
| **[Media Coverage](#media-coverage)**
| **[Call For Contribution](#call-for-contributors)** |


</p>





<h3>ProG: A Python Library for Multi-task Graph Prompting</h3> 

**ProG** (_Prompt Graph_) is a library built upon PyTorch to easily conduct single or multi-task prompting for 
pre-trained Graph Neural Networks (GNNs). The idea is derived from the paper: Xiangguo Sun, Hong Cheng, Jia Li,
etc. [All in One: Multi-task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504). KDD2023 (🔥  _**Best Research Paper Award**, which is the first time for Hong Kong and Mainland China_), in which
they released their [raw codes](https://anonymous.4open.science/r/mpg/README.md). This repository is a redesigned and enhanced version of the raw
codes with [extremely huge changes and updates](https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023)

<br>




<h3>🌟ProG++🌟: A Unified Python Library for Graph Prompting</h3> 

**ProG++** is an extended library with **ProG**, which supports more graph prompt models. Currently, **ProG++** is now in its beta version (a little baby: [ProG Plus](https://github.com/Barristen/Prog_plus)), and we will merge ``ProG Plus`` to ``ProG`` in the near future. Some implemented models are as follows (_We are now implementing more related models and we will keep integrating more models to ProG++_):  

>- [All in One] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One: Multi-Task Prompting for Graph Neural Networks,” KDD, 2023
>- [GPF Plus] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal Prompt Tuning for Graph Neural Networks,” NeurIPS, 2023.
>- [GraphPrompt] Liu Z, Yu X, Fang Y, et al. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. The Web Conference, 2023.
>- [GPPT] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks,” KDD, 2022
>- [GPF] T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt tuning for graph neural networks,” arXiv preprint, 2022.


<br>

<h3 align="center">
  
![](https://img.shields.io/badge/News-red)
🐶We released a comprehensive survey on graph prompt!

</h3>

>Xiangguo Sun, Jiawen Zhang, Xixi Wu, Hong Cheng, Yun Xiong, Jia Li.
>
>Graph Prompt Learning: A Comprehensive Survey and Beyond
>
>in arXiv [https://arxiv.org/abs/2311.16534](https://arxiv.org/abs/2311.16534)
>
>(under review in TKDE)

In this survey, we present more details of **ProG++** and also release a [repository](https://github.com/WxxShirley/Awesome-Graph-Prompt)🦀 for a comprehensive collection of research papers, benchmark datasets, and readily accessible code implementations. 


  
  **The Architecture of ProG++**

  <img height="350" src="/ProG_pipeline.jpg?sanitize=true" />
  <br>
  


**🌹Please cite our work if you find help for you:**


```
@inproceedings{sun2023all,
  title={All in One: Multi-Task Prompting for Graph Neural Networks},
  author={Sun, Xiangguo and Cheng, Hong and Li, Jia and Liu, Bo and Guan, Jihong},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'23)},
  year={2023},
  pages = {2120–2131},
  location = {Long Beach, CA, USA},
  isbn = {9798400701030},
  url = {https://doi.org/10.1145/3580305.3599256},
  doi = {10.1145/3580305.3599256}
}

@article{sun2023graph,
  title = {Graph Prompt Learning: A Comprehensive Survey and Beyond},
  author = {Sun, Xiangguo and Zhang, Jiawen and Wu, Xixi and Cheng, Hong and Xiong, Yun and Li, Jia},
  year = {2023},
  journal = {arXiv:2311.16534},
  eprint = {2311.16534},
  archiveprefix = {arxiv}
}


```

---

## Quick Start

### Package Dependencies

- PyTorch 1.13.1
- torchmetrics 0.11.4
- torch_geometric 2.2.0

### Pre-train your GNN model

The following codes present a simple example on how to pre-train a GNN model via GraphCL. You can also find a integrated
function ``pretrain()`` in ``no_meta_demo.py``.

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

The above codes are also integrated as a function ``model_create(dataname, gnn_type, num_class, task_type)`` in this
project.

### Prompt learning with hand-crafted answering template

```python
from ProG.data import multi_class_NIG
import torch

train, test, _, _ = multi_class_NIG(dataname, num_class)
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

For more detailed usage examples w.r.t ``prompt with answer tuning``, ``prompt with meta-learning`` etc. Please check
the demo in:

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

  
**Kindly note that the comparison takes the same pre-trained pth. The absolute value of performance won't mean much because the final results may vary depending on different
  pre-training states. It would be more interesting to see the relative performance with other training paradigms. **



**Note:**

- The above table is copied from this blog: https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023

## Citation

bibtex

```
@inproceedings{sun2023all,
  title={All in One: Multi-Task Prompting for Graph Neural Networks},
  author={Sun, Xiangguo and Cheng, Hong and Li, Jia and Liu, Bo and Guan, Jihong},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'23)},
  year={2023},
  pages = {2120–2131},
  location = {Long Beach, CA, USA},
  isbn = {9798400701030},
  url = {https://doi.org/10.1145/3580305.3599256},
  doi = {10.1145/3580305.3599256}
}
```
```
@article{sun2023graph,
  title = {Graph Prompt Learning: {{A}} Comprehensive Survey and Beyond},
  author = {Sun, Xiangguo and Zhang, Jiawen and Wu, Xixi and Cheng, Hong and Xiong, Yun and Li, Jia},
  year = {2023},
  journal = {arXiv:2311.16534},
  eprint = {2311.16534},
  archiveprefix = {arxiv}
}

```


## Contact

- For More Information, Further discussion, Contact: [Website](https://graphprompt.github.io/)
- Email: xiangguosun at cuhk dot edu dot hk

## Media Coverage

**Media Reports**

- [香港首位學者獲ACM頒最佳研究論文獎, 香港新聞網, 2023-09-20 15:21](http://www.hkcna.hk/docDetail.jsp?id=100474675&channel=4372)
- [内地及香港首次！港中大的他们获得这项国际大奖！,香港中文大学官方公众号， 2023-09-11 21:30](https://mp.weixin.qq.com/s/0AYazi8HD9CGRs0kxqUinw)
- [Two CUHK scholars receive Best Paper Award from ACM SIGKDD Conference 2023, CUHK Focus](https://www.focus.cuhk.edu.hk/20230906/two-cuhk-scholars-receive-best-paper-award-from-acm-sigkdd-conference-2023/)
- [Prof. Cheng Hong and her postdoc fellow Dr. Sun Xiangguo won the best paper award at KDD2023, CUHK SEEM](https://www.se.cuhk.edu.hk/prof-cheng-hong-and-her-postdoc-fellow-dr-sun-xiangguo-won-the-best-paper-award-at-kdd2023/)
- [港科夜闻｜香港科大(广州)熊辉教授、李佳教授分别荣获 ACM SIGKDD2023 服务奖与最佳论文奖(研究)](https://mp.weixin.qq.com/s/QCm-QtwNjh6rXrzJ3K2njQ)
- [数据科学与分析学域李佳教授荣获SIGKDD2023最佳论文奖（研究）！](https://mp.weixin.qq.com/s/3Efakieo9Y9Tj6DTwZoonA)
- [实时追踪科研动态丨姚期智、Quoc Viet Le等人8.9精选新论文，附ChatPaper综述](https://mp.weixin.qq.com/s/nfKiBcLIMcuvNqZT0XgSGA)
- KDD 2023奖项出炉：港中文、港科大等获最佳论文奖，GNN大牛Leskovec获创新奖
  - [机器之心](https://mp.weixin.qq.com/s/_JwfqlvFLOyauJgWxw-iWw)
  - [专知](https://mp.weixin.qq.com/s/2XLudB9BFCp8yZgLgbF3sQ)
  - [PaperWeekly](https://mp.weixin.qq.com/s/eZpMdWAG4Lg0r0EZ0O6nVA)
  - [深度学习技术前沿](https://mp.weixin.qq.com/s/PhjszSX3RGv3_Nml3dfwsQ)
  - [智源社区](https://hub.baai.ac.cn/view/28475)
- [多篇GNN论文获KDD 2023大奖, 图神经网络与推荐系统  2023-08-09 16:03](https://mp.weixin.qq.com/s/7DQC-565F8VoqLluU3WwLw)
- [港科广数据科学与分析学域李佳教授荣获SIGKDD2023最佳论文奖（研究）！](https://mp.weixin.qq.com/s/6eUT7SE6ew2N7tRCaFE6gQ)

**Online Discussion**

- [LOGS第2023/08/12期||KDD 2023 Best Paper Winner 孙相国 ：提示学习在图神经网络中的探索](https://mp.weixin.qq.com/s/vdFCNhgi2wuXscSauGbSgA)
- [Talk预告 | KDD'23 Best Paper 港中文孙相国：All in One - 提示学习在图神经网络中的探索](https://mp.weixin.qq.com/s/z8AiCwTUn2TvY8tzB4NjVg)
- [All in One Multi-Task Prompting for Graph Neural Networks 论文解读](https://www.bilibili.com/video/BV1Rk4y1V7wA/?share_source=copy_web&vd_source=dc2c6946b0127024c2225b0e695d9a83)
- [kdd2023最佳论文](https://www.bilibili.com/video/BV1Uu4y1B7zp/?share_source=copy_web&vd_source=dc2c6946b0127024c2225b0e695d9a83)
- [All in One: Multi-task Prompting for Graph Neural Networks（KDD 2023 Best Paper](https://zhuanlan.zhihu.com/p/650958869)
- [怎么评价KDD23的best paper？ - 知乎](https://www.zhihu.com/question/617300883)

**Other research papers released by us**
- [最新图大模型综述：由港科广、港中文、清华联合发布，详述使用大模型处理图任务的进展与挑战](https://mp.weixin.qq.com/s/hohAfy04rApaaqz6_3EdsQ)
- [大模型和图如何结合？最新《图遇见大型语言模型》综述，详述最新进展](https://mp.weixin.qq.com/s/maqKuu9lVqEDpSptBqwoWg)
- [香港中文领衔港科广、复旦重磅发布：迈向通用图智能的新方法，图提示学习进展与挑战](https://mp.weixin.qq.com/s/NvfgtXLUX2MWu0U2p7RKEQ)
- [香港中文领衔港科广、复旦重磅发布：迈向通用图智能的新方法，图提示学习进展与挑战](https://mp.weixin.qq.com/s/zSTFTgKGaOXbOC0kKT8raQ)
- [图上如何提示？港中文等最新《图提示学习》全面综述，详述图提示分类体系](https://mp.weixin.qq.com/s/6k7ZTVM0Hj8bO4iAjOERAQ)

## Call for Contributors!

Once you are invited as a contributor, you would be asked to follow the following steps:

- step 1. create a temp branch (e.g. ``xgTemp``) from the ``main`` branch (latest branch). 
- step 2. fetch ``origin/xgTemp`` to your local ``xgTemp``, and make your own changes via PyCharm etc.
- step 3. push your changes from local ``xgTemp`` to your github cloud branch: ``origin/xgTemp``.
- step 4. open a pull request to merge from your branch to ``main``.

When you finish all these jobs. I will get a notification and approve merging your branch to ``main``.
Once I finish, I will delete your branch, and next time you will repeat the above jobs.


A widely tested ``main`` branch will then be merged to the ``stable`` branch and a new version will be released based on ``stable`` branch.
