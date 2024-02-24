<h1 align='left'>
ProG Plus (Updating)
</h1>


<h5 align="left">

![](https://img.shields.io/badge/Latest_version-v0.1.5-red)
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/PyTorch-v2.0.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.9-red)

</h5>

<br>

🌟 ``ProG Plus`` is a baby of **ProG++**, an extended library upon [![](https://img.shields.io/badge/ProG-red)](https://github.com/sheldonresearch/ProG). ``ProG Plus`` supports more graph prompt models, and we will merge ``ProG Plus`` to [![](https://img.shields.io/badge/ProG-red)](https://github.com/sheldonresearch/ProG) in the near future (named as **ProG++**). Some implemented models are as follows (_We are now implementing more related models and we will keep integrating more models to ProG++_):  
>- ( _**KDD23 Best Paper**_ 🌟)  X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One: Multi-Task Prompting for Graph Neural Networks,” in KDD, 2023
>- M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks,” in KDD, 2022
>- T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt tuning for graph neural networks,” arXiv preprint, 2022.
>- T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal Prompt Tuning for Graph Neural Networks,” in NeurIPS, 2023.


<h5 align='center'>
  
Thanks to Dr. Xiangguo Sun for his

[![](https://img.shields.io/badge/Python_Library-ProG-red)](https://github.com/sheldonresearch/ProG)

Please visit their [website](https://github.com/sheldonresearch/ProG) to inquire more details on **ProG**, **ProG Plus**, and **ProG++**

</h5>

## TODO List

> **Note**
> <span style="color:blue"> Current experimental datasets: Node/Edge:Cora/Citeseer/Pubmed; Graph:MUTAG</span>

- [ ] **Write a comprehensive usage document**(refer to pyG)
- [ ] Dataset:  support more  graph-level datasets, PROTEINS, IMDB-BINARY, REDDIT-BINARY, ENZYMES; Add node-level datasets.
- [ ] Write a tutorial, and polish data code, to make our readers feel more easily to deal with their own data. That is to: (1) provide a demo/tutorial to let our readers know how to deal with data; (2) polish data code, making it more robust, reliable, and readable.  
- [ ] Pre_train: implementation of DGI. (Deep Graph Infomax), InfoGraph, contextpred, AttrMasking, ContextPred, GraphMAE, GraphLoG, JOAO
- [ ] Prompt: Gprompt(WWW23) prodigy(ICML23)
- [ ] induced graph(1.better way to generate induced graph/2.simplify the 3 type of generate-func)
- [x] unify args
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
| GPF       |    N , G   |
| GPPTPrompt|      N     |
| GPrompt   |   N, E, G  |
| ProGPrompt|   N,    G  |


## Environment Setup
```shell

--cuda 11.8

--python 3.9.17 

--pytorch 2.0.1 

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
## Usage

See in [https://github.com/sheldonresearch/ProG](https://github.com/sheldonresearch/ProG)

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
