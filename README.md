<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sheldonresearch/ProG">
    <img height="150" src="Logo.jpg?sanitize=true" />
  </a>
</div>

<h3 align="center">🌟ProG: A Unified Python Library for Graph Prompting🌟</h3>

<div align="center">
  
| **[Quick Start](#quick-start)** 
| **[Paper](#paper)**
| **[Media Coverage](#media-coverage)**
| **[Call For Contribution](#call-for-contributors)** |

![](https://img.shields.io/badge/Latest_version-v0.2-red)
![Testing Status](https://img.shields.io/badge/PyTorch-v1.13.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.9-red)

</div>



🌟**ProG**🌟 (Prompt Graph) is a library built upon PyTorch to easily conduct single or multi-task prompting for 
pre-trained Graph Neural Networks (GNNs). You can easily use this library to conduct various graph workflows like **supervised learning**, **pre-training and prompting**, and **pre-training and finetuning** for your node/graph-level tasks. The starting point of this library is our KDD23 paper [**All in One**](https://arxiv.org/abs/2307.01504) (Best Research Paper Award, which is the first time for Hong Kong and Mainland China).  



* The [**``ori``**](https://github.com/sheldonresearch/ProG/tree/ori) branch of this repository is the source code of [**All in One**](https://github.com/sheldonresearch/ProG/tree/ori), in which you can conduct even more kinds of tasks with more flexible graph prompts.

* The **``main``** branch of this library is the source code of [**ProG: A Graph Prompt Learning Benchmark**](https://arxiv.org/abs/2406.05346), it supports more than **5** graph prompt models (e.g. All-in-One, GPPT, GPF Plus, GPF, GraphPrompt, etc) with more than **6** pre-training strategies (e.g. DGI, GraphMAE, EdgePreGPPT, EdgePreGprompt, GraphCL, SimGRACE, etc), and have been tested on more than **15** graph datasets, covering both homophilic and heterophilic graphs from various domains with different scales.  Click [here](#supportive-list) to see the full and latest supportive list (backbones, pre-training strategies, graph prompts, and datasets). 


<div align="center">
  
**Click to See [A Full List of Our Works in Graph Prompts](#our-work)**

</div>

 
<h3 align="left">🌟Acknowledgement</h3>

<div align="left">
  
- **Leader:** [**Dr. Xiangguo SUN**](https://xgsun.mysxl.cn)
- **Consultants:** [**Prof. Jia LI**](https://sites.google.com/view/lijia), [**Prof. Hong CHENG**](https://www1.se.cuhk.edu.hk/~hcheng/)
- **Developers:** [**Mr. Chenyi ZI**](https://barristen.github.io/), [**Mr. Haihong ZHAO**](https://haihongzhao.com/), [**Dr. Xiangguo SUN**](https://xgsun.mysxl.cn)
- **Insight Suggestions:** [**Miss. Xixi WU**](https://wxxshirley.github.io) (who also contributes a lot to our [survey](https://arxiv.org/abs/2311.16534), [repository](https://github.com/WxxShirley/Awesome-Graph-Prompt), etc.)
- [**Clik Here to See Other Contributors**](https://github.com/sheldonresearch/ProG/graphs/contributors)  

</div>

Development Progress:

> - ``ori`` branch started [JUl 2023]
> -  ``main`` branch started [JUN 2024]
> -   widely testing, debugging and updating [NOW]
> -  ``stable`` branch started [reaching around 20%]

<br>

<div align="left">
  
![](https://img.shields.io/badge/Latest_News-red)
  
</div>

- **2024/10/24**: **BIG NEWS! A Detailed Hands-on Blog Coming Soon**
   > We are now trying our best to prepare a detailed, hands-on blog with deeper insights, troubleshooting, training tricks, and an entirely new perspective for graph prompting (and our ProG project). We just started recently and we plan to finish this hard work by the end of next month. Please wait for a while! 
- **2024/10/15**: We released a new work with graph prompts on cross-domain recommendation:   
   > Hengyu Zhang, Chunxu Shen, Xiangguo Sun, Jie Tan, Yu Rong, Chengzhi Piao, Hong Cheng, Lingling Yi. Adaptive Coordinators and Prompts on Heterogeneous Graphs for Cross-Domain Recommendations. [https://arxiv.org/abs/2410.11719](https://arxiv.org/abs/2410.11719)
- **2024/10/03**: We present a comprehensive theoretical analysis of graph prompt and release our theory analysis as follows:   
   > Qunzhong Wang and Xiangguo Sun and Hong Cheng. **Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis**. [https://arxiv.org/abs/2410.01635](https://arxiv.org/abs/2410.01635)
- **2024/09/26**: Our Benchmark Paper was accepted by NeurIPS 2024:   
   > Chenyi Zi, Haihong Zhao, Xiangguo Sun, Yiqing Lin, Hong Cheng, Jia Li. ProG: A Graph Prompt Learning Benchmark. [https://arxiv.org/abs/2406.05346](https://arxiv.org/abs/2406.05346)  
  - (prior news) 2024/06/08: We use our developed ProG to extensively evaluate various graph prompts, and released our analysis report as follows: Chenyi Zi, Haihong Zhao, Xiangguo Sun, Yiqing Lin, Hong Cheng, Jia Li. ProG: A Graph Prompt Learning Benchmark. [https://arxiv.org/abs/2406.05346](https://arxiv.org/abs/2406.05346)  
- **2024/01/01:** A big updated version released!
- **2023/11/28:** We released a comprehensive survey on graph prompt! 
   > Xiangguo Sun, Jiawen Zhang, Xixi Wu, Hong Cheng, Yun Xiong, Jia Li. Graph Prompt Learning: A Comprehensive Survey and Beyond [https://arxiv.org/abs/2311.16534](https://arxiv.org/abs/2311.16534)
- **2023/11/15:** We released a [🦀repository🦀](https://github.com/WxxShirley/Awesome-Graph-Prompt) for a comprehensive collection of research papers, datasets, and readily accessible code implementations. 


 <details close>
   <summary>History News</summary>
   
   - **2023/11/15:** We released a [🦀repository🦀](https://github.com/WxxShirley/Awesome-Graph-Prompt) for a comprehensive collection of research papers, datasets, and readily accessible code implementations.
</details>

<br>

## Installation
**Pypi**

From ProG 1.0 onwards, you can install and use ProG. For this, simply run
```shell
pip install prompt-graph
```
Or you can git clone our repository directly.
## Environment Setup


Before you begin, please make sure that you have Anaconda or Miniconda installed on your system. This guide assumes that you have a CUDA-enabled GPU.

```shell
# Create and activate a new Conda environment named 'ProG'
conda create -n ProG
conda activate ProG

# Install Pytorch and DGL with CUDA 11.7 support
# If your use a different CUDA version, please refer to the PyTorch and DGL websites for the appropriate versions.
conda install numpy
conda install pytorch==2.0.1 pytorch-cuda=12.2 -c pytorch -c nvidia

# Install additional dependencies
pip install torch_geometric pandas torchmetrics Deprecated 

# If you are having trouble with torch-geometric linked binary version, use conda to build it.

conda install pytorch-sparse -c pyg
```

In addition, You can use our pre-train GNN directly or use our pretrain module to pre-train the GNN you want by 
```shell
pip install torch_cluster  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
the torch and cuda version  can refer to https://data.pyg.org/whl/  

## Quick Start
The Architecture of ProG is shown as follows:

<img height="350" src="/ProG_pipeline.jpg?sanitize=true" />


Firstly, download from onedrive https://1drv.ms/u/s!ArZGDth_ySjPjkW2n-zsF3_GGvC1?e=rEnBA7 (126MB)to get `Experiment.zip`.
You can unzip to get our dataset pre-trained model which is already pre-trained, and induced graph, sample data in the few-shot setting. (Please make sure the unzipped folder's name is `/Experiment`.
if the download link is unavailable, please drop us an email to let us know(barristanzi666@gmail.com)

**Warning! The dataset providers may update dataset itself causing compatibility issues with the pretain models we provided. Reports on datasets (ENZYMES,BZR) have been found.**

**It is recommended to pretrain your model by yourself.**
``` shell
unzip Experiment.zip
```

We have provided scripts with hyper-parameter settings to get the experimental results


### With Customized Hyperparameters 
In downstream task, you can obtain the experimental results by running the parameters you want, for example, 

```shell
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/Edgepred_Gprompt.GCN.128hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GPF-plus' --shot_num 1 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0
```

```shell
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/BZR/DGI.GCN.128hidden_dim.pth' --task GraphTask --dataset_name 'BZR' --gnn_type 'GCN' --prompt_type 'All-in-one' --shot_num 1 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 1
```

### With Optimal Hyperparameters through Random Search

Perform a random search of hyperparameters for the GCN model on the Cora dataset. (NodeTask)
```shell
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth' --task NodeTask --dataset_name 'Cora' --gnn_type 'GCN' --prompt_type 'GPF-plus' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
```

<details>
  <summary ><strong>Table of The Following Contents</strong></summary>
  <ol>
     <li>
      <a href="#supportive-list">Supportive List</a>
    </li>
    <li>
      <a href="#pre-train-your-gnn-model">Pre-train your GNN model</a>
    </li>
    <li>
      <a href="#downstream-tasks">Downstream Tasks</a>
    </li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#prompt-class">Prompt Class</a></li>
    <li><a href="#environment-setup">Environment Setup</a></li>
    <li><a href="#todo-list">TODO List</a></li>
  </ol>
</details>

### with the default few-shot sample
For train and test sample split to reproduce the results in the benchmark, you can 
```unzip node.zip -d './Experiment/sample_data'```
or do not unzip use the code to split the dataset Automatically

### Supportive List

**Supportive graph prompt approaches currently (keep updating):**  

>- [All in One] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One: Multi-Task Prompting for Graph Neural Networks,” KDD, 2023
>- [GPF Plus] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal Prompt Tuning for Graph Neural Networks,” NeurIPS, 2023.
>- [GraphPrompt] Liu Z, Yu X, Fang Y, et al. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. The Web Conference, 2023.
>- [GPPT] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks,” KDD, 2022
>- [GPF] T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt tuning for graph neural networks,” arXiv preprint, 2022.



**Supportive graph pre-training strategies currently (keep updating):**  

- For node-level, we consider ``DGI`` and ``GraphMAE``, where ``DGI`` maximizes the mutual information between node and graph representations for informative embeddings and ``GraphMAE`` learns deep node representations by reconstructing masked features.
- For edge-level, we introduce ``EdgePreGPPT`` and ``EdgePreGprompt``, where ``EdgePreGPPT`` calculates the dot product as the link probability of node pairs and ``EdgePreGprompt`` samples triplets from label-free graphs to increase the similarity between the contextual subgraphs of linked pairs while decreasing the similarity of unlinked pairs.
- For graph-level, we involve ``GraphCL``, ``SimGRACE``, where ``GraphCL`` maximizes agreement between different graph augmentations to leverage structural information and ``SimGRACE`` tries to perturb the graph model parameter spaces and narrow down the gap between different perturbations for the same graph.


**Supportive graph backbone models currently (keep updating):**  

- Graph Convolutional Network (GCN), GraphSAGE, GAT, and Graph Transformer (GT).

> Beyond the above graph backbones, you can also seamlessly integrate nearly all graph models implemented by PyG.


**Click [here] to see more details information on these graph prompts, pre-training strategies, and graph backbones. **



### Pre-train your GNN model

We have designed four pre_trained classes (Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE), which is in ProG.pretrain module, you can pre_train the model by running ``pre_train.py`` and setting the parameters you want. 
Or just unzip to get our dataset pre-trained model which is already pre-trained. 
``` shell
unzip Experiment.zip
```
In the pre-train phase, you can obtain the experimental results by running the parameters you want:
```shell
python pre_train.py --task Edgepred_Gprompt --dataset_name 'PubMed' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
```

```python
import prompt_graph as ProG
from ProG.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, NodePrePrompt, GraphPrePrompt, DGI, GraphMAE
from ProG.utils import seed_everything
from ProG.utils import mkdir, get_args
from ProG.data import load4node,load4graph

args = get_args()
seed_everything(args.seed)


if args.pretrain_task == 'SimGRACE':
    pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
if args.pretrain_task == 'GraphCL':
    pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
if args.pretrain_task == 'Edgepred_GPPT':
    pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
if args.pretrain_task == 'Edgepred_Gprompt':
    pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
if args.pretrain_task == 'DGI':
    pt = DGI(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
if args.pretrain_task == 'NodeMultiGprompt':
    nonlinearity = 'prelu'
    pt = NodePrePrompt(args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3, args.device)
if args.pretrain_task == 'GraphMultiGprompt':
    nonlinearity = 'prelu'
    pt = GraphPrePrompt(graph_list, input_dim, out_dim, args.dataset_name, args.hid_dim, nonlinearity,0.9,0.9,0.1,1,0.3, 0.1, args.device)
if args.pretrain_task == 'GraphMAE':
    pt = GraphMAE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device,
                  mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2)
pt.pretrain()

```
### Load Data 
Before we do the downstream task, we need to load the nessary data. For some specific prompt, we need to choose function load_induced_graph to the input of our tasker

```python
def load_induced_graph(dataset_name, data, device):

    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
seed_everything(args.seed)

print('dataset_name', args.dataset_name)
if args.downstream_task == 'NodeTask':
    data, input_dim, output_dim = load4node(args.dataset_name)   
    data = data.to(args.device)
    if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
        graphs_list = load_induced_graph(args.dataset_name, data, args.device) 
    else:
        graphs_list = None 
         

if args.downstream_task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name)
```

### Downstream Tasks
In ``downstreamtask.py``, we designed two tasks (Node Classification, Graph Classification). Here are some examples. 
```python
import prompt_graph as ProG
from ProG.tasker import NodeTask, LinkTask, GraphTask

if args.downstream_task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name)

if args.downstream_task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer,
                    gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                    epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list)


if args.downstream_task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                    shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim)

_, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _= tasker.run()

```

  
**Kindly note that the comparison takes the same pre-trained pth. The absolute value of performance won't mean much because the final results may vary depending on different pre-training states.It would be more interesting to see the relative performance with other pre-training paradigms.**


### Bench Random Search
In our bench 
      
## Datasets

| Dataset     | Graphs | Avg.nodes | Avg.edges | Features | Node classes | Task (N / G) | Category                           |
|-------------|--------|-----------|-----------|----------|--------------|--------------|------------------------------------|
| Cora        | 1      | 2,708     | 5,429     | 1,433    | 7            | N            | Homophilic                         |
| Pubmed      | 1      | 19,717    | 88,648    | 500      | 3            | N            | Homophilic                         |
| CiteSeer    | 1      | 3,327     | 9,104     | 3,703    | 6            | N            | Homophilic                         |
| Actor       | 1      | 7600      | 30019     | 932      | 5            | N            | Heterophilic                       |
| Wisconsin   | 1      | 251       | 515       | 1703     | 5            | N            | Heterophilic                       |
| Texas       | 1      | 183       | 325       | 1703     | 5            | N            | Heterophilic                       |
| ogbn-arxiv  | 1      | 169,343   | 1,166,243 | 128      | 40           | N            | Homophilic & Large scale           |

| Dataset      | Graphs | Avg.nodes | Avg.edges | Features | Graph classes | Task (N / G) | Domain         |
|--------------|--------|-----------|-----------|----------|---------------|--------------|----------------|
| MUTAG        | 188    | 17.9      | 19.8      | 7        | 2             | G            | small molecule |
| IMDB-BINARY  | 1000   | 19.8      | 96.53     | 0        | 2             | G            | social network |
| COLLAP       | 5000   | 74.5      | 2457.8    | 0        | 3             | G            | social network |
| PROTEINS     | 1,113  | 39.1      | 72.8      | 3        | 2             | G            | proteins       |
| ENZYMES      | 600    | 32.6      | 62.1      | 18       | 6             | G            | proteins       |
| DD           | 1,178  | 284.1     | 715.7     | 89       | 2             | G            | proteins       |
| COX2         | 467    | 41.2      | 43.5      | 3        | 2             | G            | small molecule |
| BZR          | 405    | 35.8      | 38.4      | 3        | 2             | G            | small molecule |




## TODO List

> **Note**
> <span style="color:blue"> Current experimental datasets: Node/Edge:Cora/Citeseer/Pubmed; Graph:MUTAG</span>

- [ ] **Write a comprehensive usage document**(refer to pyG)
- [ ] Write a tutorial, and polish data code, to make our readers feel more easily to deal with their own data. That is to: (1) provide a demo/tutorial to let our readers know how to deal with data; (2) polish data code, making it more robust, reliable, and readable.  
- [ ] Pre_train: implementation of  InfoGraph, contextpred, AttrMasking, ContextPred, GraphLoG, JOAO
- [ ] Add Prompt: prodigy (NeurIPS'2023 Spotlight)
- [ ] induced graph(1.better way to generate induced graph/2.simplify the 3 type of generate-func)
- [ ] support deep GNN layers by adding the feature [DeepGCNLayer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html#torch_geometric.nn.models.DeepGCNLayer)


---

<a name="paper"></a>

<h3 align="center">🌹Please Cite Our Work If Helpful:</h3>
<p align="center"><strong>Thanks! / 谢谢! / ありがとう! / merci! / 감사! / Danke! / спасибо! / gracias! ...</strong></p>

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

@article{wang2024does,
      title={Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis}, 
      author={Qunzhong Wang and Xiangguo Sun and Hong Cheng},
      year={2024},
      journal = {arXiv preprint arXiv:2410.01635},
      url={https://arxiv.org/abs/2410.01635}
}


@article{zi2024prog,
      title={ProG: A Graph Prompt Learning Benchmark}, 
      author={Chenyi Zi and Haihong Zhao and Xiangguo Sun and Yiqing Lin and Hong Cheng and Jia Li},
      year={2024},
      journal = {the Thirty-Eighth Advances in Neural Information Processing Systems (NeurIPS 2024)},
      volume={},
      pages={}
}


@article{sun2023graph,
  title = {Graph Prompt Learning: A Comprehensive Survey and Beyond},
  author = {Sun, Xiangguo and Zhang, Jiawen and Wu, Xixi and Cheng, Hong and Xiong, Yun and Li, Jia},
  year = {2023},
  journal = {arXiv:2311.16534},
  eprint = {2311.16534},
  archiveprefix = {arxiv}
}

@article{zhang2024adaptive,
  title={Adaptive Coordinators and Prompts on Heterogeneous Graphs for Cross-Domain Recommendations},
  author={Hengyu Zhang and Chunxu Shen and Xiangguo Sun and Jie Tan and Yu Rong and Chengzhi Piao and Hong Cheng and Lingling Yi},
  journal={arXiv preprint arXiv:2410.11719},
  year={2024}
}

@inproceedings{li2024graph,
  title={Graph Intelligence with Large Language Models and Prompt Learning},
  author={Li, Jia and Sun, Xiangguo and Li, Yuhan and Li, Zhixun and Cheng, Hong and Yu, Jeffrey Xu},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6545--6554},
  year={2024}
}

@inproceedings{zhao2024all,
      title={All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining}, 
      author={Haihong Zhao and Aochuan Chen and Xiangguo Sun and Hong Cheng and Jia Li},
      year={2024},
      booktitle={Proceedings of the 27th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'24)}
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

@article{jin2024urban,
  title={Urban Region Pre-training and Prompting: A Graph-based Approach},
  author={Jin, Jiahui and Song, Yifan and Kan, Dong and Zhu, Haojia and Sun, Xiangguo and Li, Zhicheng and Sun, Xigang and Zhang, Jinghui},
  journal={arXiv preprint arXiv:2408.05920},
  year={2024}
}

@article{li2024survey,
      title={A Survey of Graph Meets Large Language Model: Progress and Future Directions}, 
      author={Yuhan Li and Zhixun Li and Peisong Wang and Jia Li and Xiangguo Sun and Hong Cheng and Jeffrey Xu Yu},
      year={2024},
      eprint={2311.12399},
      archivePrefix={arXiv},
      journal = {arXiv:2311.12399}
}


@article{wang2024ddiprompt,
  title={DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning},
  author={Wang, Yingying and Xiong, Yun and Wu, Xixi and Sun, Xiangguo and Zhang, Jiawei},
  journal={arXiv preprint arXiv:2402.11472},
  year={2024}
}

@inproceedings{zi2025rethinking,
  title={Rethinking Graph Prompts: Unraveling the Power of Data Manipulation in Graph Neural Networks},
  author={Zi, Chenyi and Bowen, LIU and Sun, Xiangguo and Cheng, Hong and Li, Jia},
  booktitle={ICLR 2025 (Blogpost Track)}
}

@article{zhu2025boundary,
  title={Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization},
  author={Zhu, Haojia and Jin, Jiahui and Kan, Dong and Shen, Rouxi and Wang, Ruize and Sun, Xiangguo and Zhang, Jinghui},
  journal={arXiv preprint arXiv:2503.07991},
  year={2025}
}

```

<br>


<div name="our-work",  align="center">
  
  🌟**A Full List of Our Works on Graph Prompts**🌟

  （* equal contribution † corresponding author）
  
  </div>



  

1. ![](https://img.shields.io/badge/Theory-black) Qunzhong Wang*, Xiangguo Sun*†, Hong Cheng. **Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis**. arXiv. [Paper](https://arxiv.org/abs/2410.01635)
2. ![](https://img.shields.io/badge/Framework-red) Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, Jihong Guan. **All in One: Multi-Task Prompting for Graph Neural Networks**. SIGKDD 23.  [Paper](https://arxiv.org/abs/2307.01504)
3. ![](https://img.shields.io/badge/Framework-red) Haihong Zhao*, Aochuan Chen*, Xiangguo Sun*†, Hong Cheng, Jia Li†. **All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining**. SIGKDD 24.  [Paper](https://arxiv.org/abs/2402.09834)
4. ![](https://img.shields.io/badge/Framework-red) Xi Chen, Siwei Zhang, Yun Xiong, Xixi Wu, Jiawei Zhang, Xiangguo Sun, Yao Zhang, Feng Zhao, Yulin Kang. **Prompt Learning on Temporal Interaction Graphs**. arXiv.  [Paper](https://arxiv.org/abs/2402.06326)
5. ![](https://img.shields.io/badge/Benchmark-violet) Chenyi Zi*, Haihong Zhao*, Xiangguo Sun†, Yiqing Lin, Hong Cheng, Jia Li. **ProG: A Graph Prompt Learning Benchmark**. NeurIPS 2024.  [Paper](https://arxiv.org/abs/2406.05346)
6. ![](https://img.shields.io/badge/Tutorial-brown) Xiangguo Sun, Jiawen Zhang, Xixi Wu, Hong Cheng, Yun Xiong, Jia Li. **Graph Prompt Learning: A Comprehensive Survey and Beyond**. arXiv.  [Paper](https://arxiv.org/abs/2311.16534)
7. ![](https://img.shields.io/badge/Tutorial-brown)Jia Li, Xiangguo Sun, Yuhan Li, Zhixun Li, Hong Cheng, Jeffrey Xu Yu. **Graph Intelligence with Large Language Models and Prompt Learning**. SIGKDD 24.   [Paper](https://dl.acm.org/doi/10.1145/3637528.3671456)
8. ![](https://img.shields.io/badge/Tutorial-brown) Yuhan Li*, Zhixun Li*, Peisong Wang*, Jia Li†, Xiangguo Sun, Hong Cheng, Jeffrey Xu Yu. **A Survey of Graph Meets Large Language Model: Progress and Future Directions**. IJCAI 2024.  [Paper](https://arxiv.org/abs/2311.12399)
9. ![](https://img.shields.io/badge/Blog-blue) Chenyi Zi, Bowen Liu, Xiangguo Sun, Hong Cheng, Jia Li. **Rethinking Graph Prompts: Unraveling the Power of Data Manipulation in Graph Neural Networks**. ICLR 2025 (BlogPosts). [Website](https://openreview.net/forum?id=fQtOTcZhXI)
10. ![](https://img.shields.io/badge/Application-green) Hengyu Zhang*, Chunxu Shen*, Xiangguo Sun†, Jie Tan, Yu Rong, Chengzhi Piao, Hong Cheng, Lingling Yi. **Adaptive Coordinators and Prompts on Heterogeneous Graphs for Cross-Domain Recommendations**. arXiv.  [Paper](https://arxiv.org/abs/2410.11719)
11.  ![](https://img.shields.io/badge/Application-green) Ziqi Gao, Xiangguo Sun, Zijing Liu, Yu Li, Hong Cheng, Jia Li†. **Protein Multimer Structure Prediction via PPI-guided Prompt Learning**. ICLR 2024. [Paper](https://arxiv.org/abs/2402.18813)
12. ![](https://img.shields.io/badge/Application-green) Jiahui Jin, Yifan Song, Dong Kan, Haojia Zhu, Xiangguo Sun, Zhicheng Li, Xigang Sun, Jinghui Zhang. **Urban Region Pre-training and Prompting: A Graph-based Approach**. arXiv. [Paper](https://www.arxiv.org/abs/2408.05920)
13. ![](https://img.shields.io/badge/Application-green) Yingying Wang, Yun Xiong, Xixi Wu, Xiangguo Sun, Jiawei Zhang. **DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning**. CIKM 2024. [Paper](https://arxiv.org/abs/2402.11472)
14. ![](https://img.shields.io/badge/Application-green) Haojia Zhu, Jiahui Jin, Dong Kan, Rouxi Shen, Ruize Wang, Xiangguo Sun, Jinghui Zhang. **Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization**. arXiv. [Paper](https://arxiv.org/abs/2503.07991).

















---

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

---

## Call for Contributors!

Once you are invited as a contributor, you would be asked to follow the following steps:

- step 1. create a temp branch (e.g. ``xgTemp``) from the ``main`` branch (latest branch). 
- step 2. fetch ``origin/xgTemp`` to your local ``xgTemp``, and make your own changes via PyCharm etc.
- step 3. push your changes from local ``xgTemp`` to your github cloud branch: ``origin/xgTemp``.
- step 4. open a pull request to merge from your branch to ``main``.

When you finish all these jobs. I will get a notification and approve merging your branch to ``main``.
Once I finish, I will delete your branch, and next time you will repeat the above jobs.


A widely tested ``main`` branch will then be merged to the ``stable`` branch and a new version will be released based on ``stable`` branch.



