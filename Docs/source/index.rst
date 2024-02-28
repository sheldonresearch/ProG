.. ProG_Tut documentation master file, created by
   sphinx-quickstart on Thu Feb  1 10:57:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
:github_url: https://github.com/WANGkevinCUHK/ProGTut
ProG Tutorial
=================

**ProG** *(prompt Graph)* is a library built upon `PyTorch <https://pytorch.org>`_  easily conduct single or multi-task prompting for
pre-trained Graph Neural Networks (GNNs). The idea is derived from the paper: Xiangguo Sun, Hong Cheng, Jia Li, etc. `"All in one Multi-task Prompting for Graph Neural Networks"(Sun et. al. , 2023) <https://arxiv.org/abs/2307.01504>`_ KDD2023 (**Best Research Paper Award**, *which is the first time for Hong Kong and Mainland China*), in which
they released their `raw codes <https://anonymous.4open.science/r/mpg/README.md>`_.

^^^^^^^^^^^^^^^^^^^^

.. note:: **ProG++**: A Unified Python Library for Graph Prompting

**ProG++** is an extended library with **ProG**, which supports more graph prompt models. Currently, **ProG++** is now in its beta version (a little baby: `ProG Plus <https://github.com/Barristen/Prog_plus>`_, and we will merge ``ProG Plus`` to ``ProG`` in the near future. Some implemented models are as follows (*We are now implementing more related models and we will keep integrating more models to ProG++*):
::
    [All in One] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, "All in One: Multi-Task Prompting for Graph Neural Networks," KDD, 2023

    [GPF Plus] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, "Universal Prompt Tuning for Graph Neural Networks," NeurIPS, 2023.

    [GraphPrompt] Liu Z, Yu X, Fang Y, et al. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. The Web Conference, 2023.

    [GPPT] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, "GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks," KDD, 2022

    [GPF] T. Fang, Y. Zhang, Y. Yang, and C. Wang, "Prompt tuning for graph neural networks," arXiv preprint, 2022.


.. toctree::
   :maxdepth: 1
   :caption: Install ProG

   install/installation

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/introduction

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/pre_train
   tutorial/downstream


.. toctree::
   :maxdepth: 1
   :caption: Main Package

   modules/data
   modules/evaluation
   modules/meta
   modules/model
   modules/pre_train
   modules/prompt
   modules/tasker
   modules/utils


