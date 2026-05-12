# ProG (Prompt Graph) 项目文档

## 概述

ProG 是一个图神经网络预训练和提示调优框架，支持多种数据集、预训练模型和Prompt方法。

---

## 1. 支持的数据集

### 1.1 节点级任务 (Node-Level Tasks)

| 数据集 | 描述 | 类型 |
|--------|------|------|
| `PubMed` | 生物医学文献网络 | Planetoid |
| `CiteSeer` | 学术引文网络 | Planetoid |
| `Cora` | 学术论文引用网络 | Planetoid |
| `Computers` | Amazon电子产品网络 | Amazon |
| `Photo` | Amazon照片网络 | Amazon |
| `Reddit` | 社交新闻站点 | 社交网络 |
| `WikiCS` | Wikipedia引文网络 | 知识图谱 |
| `Flickr` | 图片分享网络 | 社交网络 |
| `ogbn-arxiv` | OGB学术网络 | OGB |
| `Actor` | 演员合作网络 | WebKB |
| `Texas` | 德州大学网页网络 | WebKB |
| `Wisconsin` | 威斯康星大学网页网络 | WebKB |

### 1.2 图级任务 (Graph-Level Tasks)

| 数据集 | 描述 | 类型 |
|--------|------|------|
| `MUTAG` | 突变分子数据集 | TUDataset |
| `ENZYMES` | 蛋白质酶数据集 | TUDataset |
| `PROTEINS` | 蛋白质结构数据集 | TUDataset |
| `DD` | 蛋白质结构数据集 | TUDataset |
| `COLLAB` | 科研合作网络 | 社交网络 |
| `IMDB-BINARY` | 电影合作网络 | 社交网络 |
| `REDDIT-BINARY` | Reddit帖子网络 | 社交网络 |
| `COX2` | COX2分子数据集 | 分子图 |
| `BZR` | BZR分子数据集 | 分子图 |
| `PTC_MR` | 化学化合物数据集 | 分子图 |
| `ogbg-ppa` | OGB蛋白质关联网络 | OGB |
| `ogbg-molhiv` | OGB HIV分子数据集 | OGB |
| `ogbg-molpcba` | OGB PCBA分子数据集 | OGB |
| `ogbg-code2` | OGB代码解析数据集 | OGB |

### 1.3 链接预测任务 (Link Prediction Tasks)

- 单图场景：使用节点任务数据集
- 多图场景：使用图任务数据集

---

## 2. 支持的预训练模型 / 预训练策略

### 2.1 预训练策略类型

| 策略 | 层级 | 描述 |
|------|------|------|
| `DGI` | Node-level | Deep Graph Infomax - 最大化节点与图表示之间的互信息 |
| `GraphMAE` | Node-level | 掩码特征重构 |
| `Edgepred_GPPT` | Edge-level | 点积作为节点对之间的链接概率 |
| `Edgepred_Gprompt` | Edge-level | 三元组采样用于链接/非链接对相似性 |
| `GraphCL` | Graph-level | 最大化不同图增强之间的一致性 |
| `SimGRACE` | Graph-level | 扰动图模型参数空间 |

### 2.2 主干GNN模型

| 模型 | 描述 |
|------|------|
| `GCN` | 图卷积网络 |
| `GAT` | 图注意力网络 |
| `GraphSAGE` | 图采样聚合网络 |
| `GIN` | 图同构网络 |
| `GCov` | 图协方差网络 |
| `GraphTransformer` | 图Transformer |

---

## 3. 支持的Prompt方法

### 3.1 Prompt类型总览

| Prompt类型 | 任务支持 | 描述 |
|------------|----------|------|
| `None` | Both | 无Prompt（标准微调） |
| `GPPT` | Node/Graph | Graph Pre-Training and Prompt Tuning (KDD 2022) |
| `All-in-one` | Both | 多任务Prompting (KDD 2023 Best Paper) |
| `Gprompt` | Both | 统一预训练和下游任务的Graph Prompt |
| `GPF` | Graph | GNNs的通用Prompt调优 |
| `GPF-plus` | Graph | 增强版GPF，使用多个基函数 |

### 3.2 具体Prompt类

#### GPF Prompt (`prompt/GPF.py`)
- `GPF` - 基础Prompt调优
- `GPF_plus` - 带独立基函数的增强版GPF

#### GPPT Prompt (`prompt/GPPTPrompt.py`)
- `GPPTPrompt` - GPPT实现

#### All-in-one Prompt (`prompt/AllInOnePrompt.py`)
- `HeavyPrompt` - 带剪枝的重Prompt
- `FrontAndHead` - 前置和头部架构
- `LightPrompt` - 轻量级Prompt

#### Graph Prompt (`prompt/GPrompt.py`)
- `Gprompt` - 主要图Prompt类

#### SUPT Prompt (`prompt/SUPT.py`)
- `DiffPoolPrompt` - 可微分池化Prompt
- `SAGPoolPrompt` - 自注意力图池化Prompt

#### Multi-Graph Prompt (`prompt/MultiGprompt.py`)
- `downprompt` - 下游Prompt
- `DGIprompt` - DGI多任务Prompt
- `GraphCLprompt` - GraphCL多任务Prompt
- `Lpprompt` - 链接预测Prompt
- `featureprompt` - 特征Prompt
- `downstreamprompt` - 下游任务Prompt

---

## 4. 任务类型

| 任务类型 | 描述 |
|----------|------|
| `NodeTask` | 节点分类 |
| `GraphTask` | 图分类 |
| `LinkTask` | 链接预测 |

---

## 5. 关键源文件

| 文件路径 | 用途 |
|----------|------|
| `prompt_graph/defines.py` | 定义所有任务、数据集和Prompt类型常量 |
| `prompt_graph/data/load4data.py` | 数据集加载函数 |
| `prompt_graph/prompt/__init__.py` | Prompt模块导出 |
| `prompt_graph/pretrain/__init__.py` | 预训练模块导出 |
| `prompt_graph/model/__init__.py` | GNN模型导出 |
| `downstream_task.py` | 下游任务运行器 |
| `pre_train.py` | 预训练运行器 |

---

## 6. 使用示例

### 预训练
```bash
python pre_train.py --help
```

### 下游任务微调
```bash
python downstream_task.py --help
```

---

*文档生成时间: 2026/05/01*
