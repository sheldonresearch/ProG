<h1 align="center">
  <img height="150" src="/Logo.jpg?sanitize=true" />
</h1>





<p align="left">

![](https://img.shields.io/badge/Latest_version-v0.2-red)
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

<h3>

 ![](https://img.shields.io/badge/News-red)
  Big News！

</h3>

- We are so happy to announce that we have finished most updating works from ProG to **ProG++**! (the ``main`` branch of this repository. If you wish to find the original ProG package, go to the ``ori`` branch)
- From v0.2, the term "ProG" means ProG++ by default!

---


<h3 align="center">🌟ProG🌟: A Unified Python Library for Graph Prompting</h3> 

**ProG** (_Prompt Graph_) is a library built upon PyTorch to easily conduct single or multi-task prompting for 
pre-trained Graph Neural Networks (GNNs). The idea is derived from the paper: Xiangguo Sun, Hong Cheng, Jia Li,
etc. [All in One: Multi-task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504). KDD2023 (🔥  _**Best Research Paper Award**, which is the first time for Hong Kong and Mainland China_)

**ProG++** (the ``main`` branch of this repository) is an extended library of the original ``ProG`` (see in the ``ori`` branch of this repository), which supports more graph prompt models. Some implemented models are as follows (_We are now implementing more related models and we will keep integrating more models to ProG++_):  

>- [All in One] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One: Multi-Task Prompting for Graph Neural Networks,” KDD, 2023
>- [GPF Plus] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal Prompt Tuning for Graph Neural Networks,” NeurIPS, 2023.
>- [GraphPrompt] Liu Z, Yu X, Fang Y, et al. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. The Web Conference, 2023.
>- [GPPT] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks,” KDD, 2022
>- [GPF] T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt tuning for graph neural networks,” arXiv preprint, 2022.

**From now on (v0.2), the term "ProG" means ProG++ by default!**

<br>

<h3>

We released a comprehensive survey on graph prompt!

</h3>

>Xiangguo Sun, Jiawen Zhang, Xixi Wu, Hong Cheng, Yun Xiong, Jia Li.
>
>Graph Prompt Learning: A Comprehensive Survey and Beyond
>
>in arXiv [https://arxiv.org/abs/2311.16534](https://arxiv.org/abs/2311.16534)
>


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


@article{zhao2024all,
      title={All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining}, 
      author={Haihong Zhao and Aochuan Chen and Xiangguo Sun and Hong Cheng and Jia Li},
      year={2024},
      eprint={2402.09834},
      archivePrefix={arXiv}
}


@inproceedings{gao2024protein,
  title={Protein Multimer Structure Prediction via {PPI}-guided Prompt Learning},
  author={Ziqi Gao and Xiangguo Sun and Zijing Liu and Yu Li and Hong Cheng and Jia Li},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://openreview.net/forum?id=OHpvivXrQr}
}


@article{chen2024prompt,
      title={Prompt Learning on Temporal Interaction Graphs}, 
      author={Xi Chen and Siwei Zhang and Yun Xiong and Xixi Wu and Jiawei Zhang and Xiangguo Sun and Yao Zhang and Yinglong Zhao and Yulin Kang},
      year={2024},
      eprint={2402.06326},
      archivePrefix={arXiv},
      journal = {arXiv:2402.06326}
}

@article{li2024survey,
      title={A Survey of Graph Meets Large Language Model: Progress and Future Directions}, 
      author={Yuhan Li and Zhixun Li and Peisong Wang and Jia Li and Xiangguo Sun and Hong Cheng and Jeffrey Xu Yu},
      year={2024},
      eprint={2311.12399},
      archivePrefix={arXiv},
      journal = {arXiv:2311.12399}
}

```

---

## Quick Start
We have provided scripts with hyper-parameter settings to get the experimental results

In the pre-train phase, you can obtain the experimental results by running the parameters you want:
```shell
python pre_train.py --task Edgepred_Gprompt --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
```
or run `pre_train.sh`
```shell
cd scripts
./ pre_train.sh
```
In downstream_task, you can obtain the experimental results by running the parameters you want:

```shell
python downstream_task.py --pre_train_path 'None' --task GraphTask --dataset_name 'MUTAG' --gnn_type 'GCN' --prompt_type 'None' --shot_num 10 --hid_dim 128 --num_layer 3 --epochs 50 --seed 42 --device 5
```
or run `GraphTask.sh` for Graph task in **MUTAG** dataset, or run run `NodeTask.sh` for Node task in **Cora** dataset.




### Pre-train your GNN model

We have designed four pre_trained class (Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE), which is in ProG.pretrain module, you can pre_train the model by running ``pre_train.py`` and setting the parameters you want.

```python
import prompt_graph as ProG
from ProG.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
from ProG.utils import seed_everything
from ProG.utils import mkdir, get_args


args = get_args()
seed_everything(args.seed)
mkdir('./pre_trained_gnn/')

if args.task == 'SimGRACE':
    pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'GraphCL':
    pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'Edgepred_GPPT':
    pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
if args.task == 'Edgepred_Gprompt':
    pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)

pt.pretrain()



```
### Do the Downstreamtask
In ``downstreamtask.py``, we designed two tasks (Node Classification, Graph Classification). Here are some examples. 
```python
import prompt_graph as ProG
from ProG.tasker import NodeTask, LinkTask, GraphTask

if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_GPPT.GCN.128hidden_dim.pth', 
                    dataset_name = 'Cora', num_layer = 3, gnn_type = 'GCN', prompt_type = 'None', epochs = 150, shot_num = 5)
    tasker.run()


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = './pre_trained_gnn/MUTAG.SimGRACE.GCN.128hidden_dim.pth', 
                    dataset_name = 'MUTAG', num_layer = 3, gnn_type = 'GCN', prompt_type = 'All-in-one', epochs = 150, shot_num = 5)
    tasker.run()

```



  
**Kindly note that the comparison takes the same pre-trained pth.The absolute value of performance won't mean much because the final results may vary depending on different
  pre-training states.It would be more interesting to see the relative performance with other training paradigms.**





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




## TODO List

> **Note**
> <span style="color:blue"> Current experimental datasets: Node/Edge:Cora/Citeseer/Pubmed; Graph:MUTAG</span>

- [ ] **Write a comprehensive usage document**(refer to pyG)
- [ ] Write a tutorial, and polish data code, to make our readers feel more easily to deal with their own data. That is to: (1) provide a demo/tutorial to let our readers know how to deal with data; (2) polish data code, making it more robust, reliable, and readable.  
- [ ] Pre_train: implementation of DGI. (Deep Graph Infomax), InfoGraph, contextpred, AttrMasking, ContextPred, GraphMAE, GraphLoG, JOAO
- [ ] Debug Gprompt inference, All-in-one Tune，graphcl loss
- [ ] Add Prompt: prodigy (NeurIPS'2023 Spotlight)
- [ ] induced graph(1.better way to generate induced graph/2.simplify the 3 type of generate-func)
- [ ] add prompt type table (prompt_type, prompt paradigm, loss function, task_type)
- [ ] add pre_train type table
- [ ] support deep GNN layers by adding the feature [DeepGCNLayer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html#torch_geometric.nn.models.DeepGCNLayer)
      
## Dataset

| Graphs    | Graph classes | Avg. nodes | Avg. edges | Node features | Node classes | Task (N/E/G) |
|-----------|---------------|------------|------------|---------------|--------------|------------|
| Cora      | 1             | 2,708      | 5,429      | 1,433         | 7            |N           |
| Pubmed    | 1             |19,717      | 88,648     | 500           | 3            |N           |
| CiteSeer  | 1             | 3,327      | 9,104      | 3,703         | 6            |N           |
| Mutag     | 188           | 17.9       | 39.6       | ?             | 7            |N           |
| Reddit    | 1             | 232,965    | 23,213,838 | 602           | 41           |N           |
| Amazon    | 1             | 13,752     | 491,722    | 767           | 10           |N           |
| [Flickr](https://snap.stanford.edu/data/web-flickr.html)    | 1             | 89,250     | 899,756    | 500           | 7            | N          |
| PROTEINS  | 1,113         | 39.06      | 72.82      | 1             | 3            | N, G       |
| ENZYMES   | 600           | 32.63      | 62.14      | 18            | 3            | N, G       |

## Prompt Class
| Graphs    | Task (N/E/G)|
|-----------|------------|
| GPF       |      G     |
| GPPTPrompt|      N     |
| GPrompt   |   N, E, G  |
| ProGPrompt|   N,    G  |


## Environment Setup
```shell

--Python 3.9.17 

--PyTorch 2.0.1 

--torch-geometric 2.3.1

```

installation for PYG **[quick start](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)**

```shell
pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html # Optional dependencies

```
or run this command
```shell
conda install pyg -c pyg
```
