# 数据集

本文档汇总 ProG 支持的数据集、它们的来源、磁盘根目录约定，以及一些 dataset 特有的处理（容易踩坑的地方放在 §5）。

> 入口函数：`prompt_graph.data.load4node(name)` / `load4graph(name)`，分别返回 `(data, input_dim, output_dim)`（node）或 `(input_dim, output_dim, dataset_or_graph_list)`（graph）。  
> 所有数据集名称集中在 `prompt_graph/defines.py:NODE_TASKS / GRAPH_TASKS`。

---

## 1. 节点级任务（NodeTask）

`load4node` 支持的数据集与底层 loader：

| 数据集 | 来源 | torch_geometric 类 | 备注 |
|---|---|---|---|
| `Cora`, `CiteSeer`, `PubMed` | Planetoid | `Planetoid` | `NormalizeFeatures()` 已套；落地 `<DATA_ROOT>/Planetoid/<name>/` |
| `Computers`, `Photo` | Amazon Co-purchase | `Amazon` | `<DATA_ROOT>/amazon/<name>/` |
| `Reddit` | Reddit | `Reddit` | `<DATA_ROOT>/Reddit/` |
| `WikiCS` | WikiCS | `WikiCS` | `<DATA_ROOT>/WikiCS/` |
| `Flickr` | Flickr | `Flickr` | `<DATA_ROOT>/Flickr/` |
| `Wisconsin`, `Texas` | WebKB | `WebKB` | `<DATA_ROOT>/<name>/` |
| `Actor` | Actor co-occurrence | `Actor` | `<DATA_ROOT>/Actor/` |
| `ogbn-arxiv` | OGB | `PygNodePropPredDataset` | 走 `ogb_torch_load_compat()` 兼容 torch≥2.4；`<DATA_ROOT>/ogbn_arxiv/`（实际由 OGB 决定子目录） |
| `ENZYMES`, `PROTEINS` | TUDataset | `TUDataset` | 实际是多图 → 节点分类；详见 §5.1 |

未来要新增节点级数据集，至少改两处：
1. `prompt_graph/data/load4data.py:load4node` 加分支；
2. `prompt_graph/defines.py:NODE_TASKS` 加名字。

---

## 2. 图级任务（GraphTask）

`load4graph` 支持的数据集：

| 数据集 | 来源 | 备注 |
|---|---|---|
| `MUTAG`, `ENZYMES`, `COLLAB`, `PROTEINS`, `IMDB-BINARY`, `REDDIT-BINARY`, `COX2`, `BZR`, `PTC_MR`, `DD` | TUDataset | `<DATA_ROOT>/TUDataset/<name>/`（路径来自 `paths.tudataset_root()`） |
| `ogbg-ppa`, `ogbg-molhiv`, `ogbg-molpcba`, `ogbg-code2` | OGB | `<DATA_ROOT>/ogbg/<name>/`；走 `ogb_torch_load_compat()` |

`COLLAB`、`IMDB-BINARY`、`REDDIT-BINARY` 默认没有节点特征，loader 会用节点度作为特征（`node_degree_as_features`）。OGB graph 数据集统一这样处理。

`load4graph(name, pretrained=True)` 返回 `(input_dim, out_dim, graph_list)`（python list），否则返回 `(input_dim, out_dim, dataset)`（pyg Dataset）。下游 task 用 list 形式，预训练用 Dataset 形式。

---

## 3. 磁盘根目录与环境变量

默认根目录是 `<repo>/data` 和 `<repo>/Experiment`。常用环境变量见 [`Docs/architecture.md`](./architecture.md#5-数据--实验--ogb-路径)：

```bash
# 想把数据集放到 /mnt/datasets：
export PROG_DATA_ROOT=/mnt/datasets

# OGB 旧版默认 ./dataset，新版默认 <DATA_ROOT>/OGB。
# 升级 ProG 时如果不想重新下载，指回旧路径：
export PROG_OGB_ROOT=$(pwd)/dataset

# 实验产物（少样本切分 / 预训练权重 / Excel）单独存放：
export PROG_EXPERIMENT_ROOT=$(pwd)/Experiment
```

代码里**不要**自己拼路径，调 `prompt_graph.utils.paths` 的工厂函数（`induced_graph_dir` / `sample_dir` / `tudataset_root` / `ogb_dataset_root` 等）。

---

## 4. 少样本切分

下游任务统一用 k-shot：每类挑 k 个训练样本，其余作为测试。逻辑见 `load4data.py:node_sample_and_save` / `graph_sample_and_save`。

切分结果落盘到 `EXPERIMENT_ROOT/sample_data/<task>/<dataset>/<k>_shot/<task_id>/`，单个 fold 包含：

```
train_idx.pt   train_labels.pt
test_idx.pt    test_labels.pt
```

`NodeTask` / `GraphTask` 在 `create_few_data_folder` 阶段写入；同名目录已存在则不重新切（避免 reproducibility 漂移）。如果想重新切，删掉对应 `<dataset>/<k>_shot/<task_id>/` 子目录。

> `node_sample_and_save` 的"测试集"实际是从全集随机抽 90%（小于 1000 节点的数据集是 70%），不是论文里的固定切分。如果你想沿用标准切分，请绕开这个 helper 直接用 `data.train_mask` / `data.test_mask`。

---

## 5. 特殊处理（容易踩坑的部分）

### 5.1 `ENZYMES`、`PROTEINS` 作为 NodeTask

`ENZYMES` 和 `PROTEINS` 原本是 TUDataset 多图数据集，但 `load4node` 把它们当**节点分类**用：

```python
elif dataname in ['ENZYMES', 'PROTEINS']:
    dataset = TUDataset(root=str(tudataset_root()), name=dataname, use_node_attr=True)
    node_class = dataset.data.x[:, -3:]            # 节点 onehot 标签藏在 x 末三列
    input_dim = dataset.num_node_features
    out_dim = dataset.num_node_labels
    data = Batch.from_data_list(dataset)            # 把所有图合成一个大图
    data.y = node_class.nonzero().T[1]
```

也就是说：节点特征的最后 3 列被解读成 one-hot label，剩下的列才是真正的 `x`。这个约定来自原始 TUDataset，但用 `load4node('ENZYMES')` 时**不会自动 strip 那三列**——`input_dim = dataset.num_node_features` 仍然包含它们。要修改这一行为请同时改这个分支和 §1 的 NodeTask 文档。

### 5.2 `ogbn-arxiv` 标签 squeeze

OGB 节点分类的 `y` 形状是 `[N, 1]`，但 ProG 的下游任务期望 `[N]`。`NodeTask.__init__` 显式 squeeze：

```python
if self.dataset_name == 'ogbn-arxiv':
    self.data.y = self.data.y.squeeze()
```

如果你引入新的 OGB 节点级数据集，可能也要做这步。

### 5.3 OGB + torch ≥ 2.4 的 `weights_only` 兼容

OGB 内部仍然用 `torch.load(...)` 不带 `weights_only=False`，但 torch ≥ 2.4 默认 `weights_only=True`，会拒绝加载 OGB 写的 `.pt`。

`load4data.ogb_torch_load_compat()` 是一个 context manager，临时把 `torch.load` 的默认 `weights_only` 切到 `False`：

```python
with ogb_torch_load_compat():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=...)
```

实现做了线程安全 + 防止 nested 重入误重置。**新加 OGB 数据集时必须套这个 with**。一旦 OGB 自己 fix 了，这个 shim 可以删。

### 5.4 `COLLAB` / `IMDB-BINARY` / `REDDIT-BINARY` 没有节点特征

`load4graph` 检测到这三个数据集时自动调 `node_degree_as_features`，把节点度数加进 `x`。新加无 `x` 的图数据集时记得复用 `node_degree_as_features`。

### 5.5 `Texas` + GPPT 的特殊维度

`Texas` 只有 5 个 class，但 GPPT 的初始化里把 `output_dim` 同时当 num_tokens 用。在 `tasker/task.py:initialize_prompt` 有硬编码：

```python
if self.prompt_type == 'GPPT' and self.dataset_name == 'Texas':
    self.prompt = GPPTPrompt(self.hid_dim, 5, self.output_dim, device=...)
```

新加小类数 dataset + GPPT 时注意这个分支。

---

## 6. 不在支持列表的数据集

ProG 现在**不**支持：

- Link prediction 的下游任务（`load4link_prediction_*` 存在，但 `LinkTask` 没接入 strategy 框架，参看 [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) §1.10）；
- TUDataset 之外的、需要节点级标签的多图数据集；
- 异质图（`HeteroData`）。

新加这些会牵涉到 `BaseTask` 的若干隐式假设，建议先在 IMPROVEMENTS.md 开一个 follow-up section。

---

## 7. 相关文档

- [`Docs/architecture.md`](./architecture.md) — 模块边界与初始化顺序；
- [`Docs/running.md`](./running.md) — CLI 怎么传 dataset；
- [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) — 历史 bug 与 follow-up 列表；
- `prompt_graph/data/load4data.py` — 权威实现。
