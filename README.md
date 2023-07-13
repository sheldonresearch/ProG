# ProG

ProG (Prompt Graph) is a library built upon PyTorch to easily conduct single or multiple task prompting for a pre-trained Graph Neural Networks (GNNs). The idea is derived from the paper: Xiangguo Sun, Hong Cheng, JIa Li,
etc. [All in One: Multi-task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504). KDD2023, in which they released the raw
codes at [Click](https://anonymous.4open.science/r/mpg/README.md). This repository is a **polished version** of the raw codes
with **[Extremely Huge Changes and Updates](https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023)**.

- [Historical Update Logs](https://github.com/sheldonresearch/ProG/blob/main/History.md)
- [Historical Releases](https://github.com/sheldonresearch/ProG/releases)
- [Differences](https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023)


**Usage examples can be found in:**

- ``no_meta_demo.py``
- ``meta_demo.py``

For More Information, see the homepage of this paper: [Website](https://graphprompt.github.io/)

citation bibtex

```
@inproceedings{sun2023all,
  title={All in One: Multi-Task Prompting for Graph Neural Networks},
  author={Sun, Xiangguo and Cheng, Hong and Li, Jia and Liu, Bo and Guan, Jihong},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'23)},
  year={2023}
}

```

**Compare this new implementation with the raw codes:**

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

The above table is copied from this blog: https://github.com/sheldonresearch/ProG/blob/main/History.md#13-jul-2023

