# ProG 项目架构与代码改进建议

> 阅读范围：`bench.py`、`pre_train.py`、`downstream_task.py`、`create_excel_for_bench.py`、`prompt_graph/` 整个包（`data`、`model`、`pretrain`、`prompt`、`tasker`、`evaluation`、`utils`、`defines.py`）以及顶层配置文件。本文档聚焦“需要改进的地方”，给出具体的文件、行号、影响和建议，不对一次重写做包大揽。

文档结构：
1. [关键 Bug（建议优先修复）](#1-关键-bug建议优先修复)
2. [架构层面的问题](#2-架构层面的问题)
3. [代码质量问题](#3-代码质量问题)
4. [工程实践问题](#4-工程实践问题)
5. [文档与仓库结构](#5-文档与仓库结构)
6. [建议的落地路线](#6-建议的落地路线)

---

## 1. 关键 Bug（建议优先修复）

这一节列出会直接影响正确性或导致运行报错的问题，建议在做任何重构之前先修。

### 1.1 `GraphTask.GpromptTrain` 在第一个 batch 就 return（功能性 Bug）

`prompt_graph/tasker/graph_task.py:121-152` —— `return total_loss / len(train_loader), mean_centers` 被错误地写在了 `for batch in train_loader:` 循环内部，每个 epoch 只会用掉一个 batch 就 return。对比 `node_task.py:178-204` 中正确实现（return 在循环外），这显然是缩进失误。

**影响**：图任务下使用 Gprompt 的训练结果不可信，远未充分利用数据。

**建议**：把 `return` 退一级缩进。

### 1.2 `pre_train.py` 中 MultiGprompt 分支引用未定义变量 ✅ 已修

> **状态**：已修（commit `647d6c4` `fix(pretrain): wire up GraphMultiGprompt using load4graph and GraphPrePrompt`）。`pre_train.py:80-115` 现在走 `load4graph(args.dataset_name, pretrained=True)` → `GraphPrePrompt(...)` 的真实路径，分支条件覆盖 `NodeMultiGprompt` / `MultiGprompt` / `GraphMultiGprompt` 三种 task 名。

原报告：

`pre_train.py:33-37`：
```python
elif args.pretrain_task == 'GraphMultiGprompt' or args.dataset_name in GRAPH_TASKS:
    #TODO: Bugged unknown parameters: graph_list, input_dim, out_dim
    nonlinearity = 'prelu'
    #graph_list, input_dim, out_dim = load4graph(args.dataset_name, pretrained=True)
    pt = GraphPrePrompt(graph_list, input_dim, out_dim, ...)
```
`load4graph` 调用被注释掉，下面却继续引用 `graph_list / input_dim / out_dim`。任何走到这条分支的命令都会 `NameError`。

### 1.3 `BaseTask.return_pre_train_type` 在没有命中名字时返回 `None`

`prompt_graph/tasker/task.py:141-145`：
```python
def return_pre_train_type(self, pre_train_model_path):
    names = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
    for name in names:
        if name in pre_train_model_path:
            return name
```
没有最终 `return`。`bench.py:139` 拿到的 `pre_train_type=None` 后会一路写进 Excel 列名（`f"{None}+{prompt_type}"`），看上去“正常”但实际上掩盖了路径不规范的问题。同时 `'None'` 这个字符串作为名字本身就是反模式——`'GraphCL' in '...None...'` 也可能误命中。

**建议**：用更可靠的解析，例如根据 `os.path.basename` 取首段，对集合做 exact match，匹配不到时显式抛错。

### 1.4 `Experiment/sample_data/Node/...` 总是用 `task_num=5` 硬编码生成 5 份采样

`node_task.py:38-47`：
```python
for k in range(1, task_num+1):
    k_shot_folder = './Experiment/sample_data/Node/' + ...
    for i in range(1, task_num+1):
```
外层 `for k in range(1, task_num+1)` 直接覆盖了入参 `k = self.shot_num`，所以每次跑都会顺带把 `1_shot ~ 5_shot` 全部目录都生成。后续 `run()` 又只用 `self.shot_num` 那一份。

**影响**：第一次冷启动会生成大量没必要的数据；调试时也容易看错。

**建议**：去掉对 `k` 的二次循环，或者只为 `self.shot_num` 生成。`graph_task.py:31-43` 也有同样的问题。

### 1.5 `split_induced_graphs` 自带“测试数据泄漏”注释

`prompt_graph/data/induced_graph.py:80`：
```python
pos_nodes = torch.argwhere(data.y == int(current_label))   # Test data may leak
```
当子图太小时会用同标签的其他节点补齐，但选择候选时没有排除测试集节点。注释证明作者已经知情但没修。few-shot 节点分类的测试集和训练集都包含在内，会让评估指标偏高。

**建议**：传入 `train_mask` 或 `idx_train`，从候选集中过滤掉非训练节点；或者干脆只用结构信息而不补节点。

### 1.6 `bench.py` 设备分配不一致

`bench.py:16-21` 通过 `get_runtime_device(args.device)` 计算了 `runtime_device`（支持 CUDA/MPS/CPU），并用它把 `data` 搬到 device 上（109 行）。但接着把整数 `args.device` 直接传给 `NodeTask`/`GraphTask`（128/135 行），`BaseTask` 里再次根据环境变量推 device。

**影响**：当 `PROG_USE_MPS=1` 但 `args.device=0` 时，data 走 MPS，模型也走 MPS，巧合下能跑；改一下条件就会 data 在 mps 而 model 在 cpu。

**建议**：把 `runtime_device` 直接传进 `NodeTask`/`GraphTask`，下层不再自己推断；或者全局只在 `seed_everything` 处统一一次。

### 1.7 `prompt_graph/utils/seed.py` 中 `seed_torch` 重复存在

`utils/seed.py` 里同时有 `seed_everything`（行 9）和 `seed_torch`（行 38），逻辑几乎一样，只是缩进风格不同（一个用空格、一个用 tab）。`__init__.py` 只导出 `seed_everything`，`seed_torch` 没人引用，是死代码。

**建议**：删除 `seed_torch`，统一用 `seed_everything`。

### 1.8 `utils/process.py` 使用了 `np.bool`，新版 numpy 会报错

`utils/process.py:131`：
```python
return np.array(mask, dtype=np.bool)
```
NumPy ≥ 1.20 起 `np.bool` 已被弃用，在 1.24 中被移除。一旦 MultiGprompt 路径走到 `sample_mask`，会直接 `AttributeError`。

**建议**：改为 `dtype=bool`。

### 1.9 `bench.py` 超参网格存在不合理的反向区间

`bench.py:60`：
```python
'weight_decay': 10 ** np.linspace(-5, -6, 1000),
```
从 `1e-5` 递减到 `1e-6`，本身没错，但配合 `np.linspace(..., 1000)` 用作离散网格其实点都很密集（1000 个点），而 `'batch_size': np.linspace(512, 512, 200)` 产生 200 个相同的 512，完全没有意义。这暴露超参搜索没有真正“搜”而是“随机取”。

**建议**：要么显式离散网格 + `random.choice`，要么换成 Optuna/Ray Tune；现有写法实际等价于固定值，跑 `num_iter` 次只是不同 lr/wd 的随机点。

### 1.10 `LinkTask` 已基本失修 ✅ 已从公共 API 移除

> **状态**：commit `e76e20f` (`chore: drop LinkTask from public API and expose --num_iter CLI arg`) 把 `LinkTask` 从 `prompt_graph/tasker/__init__.py` 的 `__all__` 中移除，`from prompt_graph.tasker import LinkTask` 不再可用；`link_task.py` 文件本身保留作为未来重做的起点。若要复活，需按下面列的清单做。

`prompt_graph/tasker/link_task.py`：
- 数据集硬编码为 `Planetoid('data/Planetoid', name='Cora')`，完全忽略 `args.dataset_name`。
- 不支持任何 `prompt_type`，只跑 GNN。
- `epochs` 硬编码 101。

**复活清单**：接 `args.dataset_name`、走 `load4link_prediction_*`、支持 `prompt_type`、改写为 `PromptStrategy` 实现、再重新 export。

---

## 2. 架构层面的问题

### 2.1 GNN 模型严重重复（6 份 90% 雷同代码）

`prompt_graph/model/{GAT,GCN,GIN,GraphSAGE,GCov,GraphTransformer}.py` 共 6 个文件，每个 ~90 行，**实质差异只有一行**：

```python
GraphConv = GATConv          # GAT.py
GraphConv = GCNConv          # GCN.py
GraphConv = SAGEConv         # GraphSAGE.py
GraphConv = TransformerConv  # GraphTransformer.py
GraphConv = GConv            # GCov.py
GraphConv = lambda i, h: GINConv(nn.Sequential(...))  # GIN.py
```

其余 `forward`、`decode`、`pool` 选择、JK 拼接逻辑全部一致。每个文件还都重复了一份相同的 import 列表（包含大量没用到的 `import sklearn`/`import gc`/`import numpy`），看起来像是一开始 copy 出来后没回头清理。

**建议**：
- 统一成单文件 `model/gnn.py`，提供 `class GNN(nn.Module)`，构造参数里加 `conv_type: str`。
- 用 `GNN_REGISTRY = {'GCN': GCNConv, 'GAT': GATConv, ...}` 注册卷积。GIN 的工厂闭包单列。
- `__init__.py` 提供 `from .gnn import GNN as GCN/GAT/...` 的别名，保留旧 import 兼容性。

预期收益：删 500+ 行重复代码、新增一个 GNN 只需要往 registry 加一行。

### 2.2 GNN 工厂逻辑在 `BaseTask` 和 `PreTrain` 各写了一份

`prompt_graph/tasker/task.py:113-128` 和 `prompt_graph/pretrain/base.py:26-43`：完全一样的 `if gnn_type == 'GAT': ... elif gnn_type == 'GCN': ...` 链。

**建议**：把 2.1 提到的 registry 抽到 `prompt_graph.model.factory.build_gnn(name, **kwargs)`，所有调用方共用。

### 2.3 设备选择逻辑在四处重复

- `bench.py:16-21`（`get_runtime_device`）
- `tasker/task.py:19-24`（`BaseTask.__init__`）
- `pretrain/base.py:9-14`（`PreTrain.__init__`）
- `pretrain/MultiGPrompt.py:17-22, 228-229`（两遍）

四套逻辑还有微妙差异：`bench.py` 的 cpu 分支是 fallthrough，`MultiGPrompt` 的甚至嵌套了多个 `if torch.cuda.is_available()`。

**建议**：抽到 `prompt_graph/utils/device.py`，提供：
```python
def get_device(device: int | str | None = None) -> torch.device: ...
```
- 接受 `int` 时按 CUDA 索引解释；
- 接受 `'mps'`、`'cpu'`、`'cuda:N'` 字符串；
- 支持 `'auto'`，按 CUDA > MPS > CPU 探测。

把 `PROG_USE_MPS` 环境变量这种隐式接口换成 CLI 参数 `--device {auto|cuda:0|mps|cpu}`，更可发现。

### 2.4 Tasker 是 1000+ 行的“上帝类”

`graph_task.py` 26995 字节、`node_task.py` 22949 字节，单个 `run()` 方法都接近 200 行。它们做的事情完全一样：
1. 准备 sample data；
2. 初始化 GNN / answering / prompt / optimizer；
3. 在 `prompt_type` 上分发 5~6 个不同的训练函数；
4. 跑 epoch 循环 + early stopping；
5. 在 `prompt_type` 上再分发一次评估函数；
6. 聚合 metrics。

这暴露了**抽象的缺失**：每个 `prompt_type` 是一个完整的"训练-评估"策略，但被零散地塞进 if/elif。新增一个 prompt 需要改十几个文件，删一个需要在 train、eval、optimizer init、prompt init 四个地方搜代码。

**建议**：引入 Strategy 接口：

```python
class PromptStrategy(Protocol):
    def setup(self, task: BaseTask) -> None: ...
    def configure_optimizer(self, task: BaseTask) -> Optimizer: ...
    def train_epoch(self, task: BaseTask, batch_or_data) -> float: ...
    def evaluate(self, task: BaseTask, batch_or_data) -> Metrics: ...
```

每个 prompt 类型独立成一个类（`AllInOneStrategy`、`GprompptStrategy`、`GPPTStrategy`...），`BaseTask` 只持有 `strategy: PromptStrategy`。`graph_task` 和 `node_task` 的 `run()` 收敛到 ~50 行。

### 2.5 `BaseTask` 与子类之间的初始化顺序耦合

`GraphTask.__init__` 行 22-29 在 `super().__init__()` 之后才设置 `self.input_dim` / `self.output_dim`，但 `super().__init__()` 里调用了 `initialize_lossfn()`，未来如果再加入依赖 `input_dim` 的初始化就会出问题；`NodeTask.__init__` 又把 `gnn`/`prompt` 的初始化推迟到 `run()` 里。两个子类的"什么在 `__init__` 里、什么在 `run()` 里"的边界不一致，阅读者必须每次跳到具体类才能知道。

**建议**：在 `BaseTask` 里定义清晰的钩子顺序，例如：
```
__init__ -> _build_dataset() -> _build_model() -> _build_optimizer()
run     -> for task in tasks: _train() -> _evaluate()
```
子类只 override 钩子，不再自由发挥。

### 2.6 数据加载和路径管理耦合

`load4data.py`：
- `get_data_root()` 一头是相对路径计算 `os.path.dirname(__file__) ... 'data'`，但 `load4node` 里 `ENZYMES/PROTEINS` 那个分支（行 238）和 `load4link_prediction_multi_graph`（行 273）又写成了 `root='data/TUDataset'`，完全绕过 `get_data_root()`。
- `PygGraphPropPredDataset` 在 `load4link_prediction_*`（行 276、317）里 root 直接是 `'./dataset'`，和顶层 `data/` 不一致。
- 顶层脚本依赖工作目录在仓库根目录运行（`./Experiment/...`），从其他目录调用就会拿不到数据。

**建议**：
- 引入 `PROG_DATA_ROOT` / `PROG_EXPERIMENT_ROOT` 环境变量，默认指向仓库根目录下的 `data/`、`Experiment/`。
- 所有路径都通过 `prompt_graph.utils.paths` 集中管理；不允许业务代码出现裸字符串路径。

### 2.7 `torch.load` monkey-patch（`ogb_torch_load_compat`）

`load4data.py:20-32` 用 contextmanager 把 `torch.load` 替换成默认 `weights_only=False` 的版本。OGB 内部加载 .pt 时为了通过 PyTorch 2.4+ 的安全默认值。

**风险**：
- 在 with 块内任何 Python 线程的 `torch.load` 都会受影响（包括 DataLoader workers）。
- 一旦异常路径绕过 `finally`，全局 `torch.load` 永久带 `weights_only=False`，等于关掉了安全检查。

**建议**：直接 fork 一份调用 OGB 的逻辑，传 `weights_only=False`；或者用 `unittest.mock.patch` 显式作用域，至少加 `threading.RLock`。

### 2.8 任务划分逻辑分散且语义不一致

- `node_sample_and_save` (`load4data.py:34`)：先随机取 90% 测试，再从 10% 候选里选 k-shot。
- `graph_sample_and_save` (`load4data.py:61`)：先随机取 80% 测试，再用每类前 k 个，**没有 shuffle 候选集，直接 `class_indices[:k]`**，结果不随机。
- `graph_split` (`graph_split.py`)：又是另一套（90/10/5 切分），`val_dataset_size = remaining // 9` 这种比例既不是 8:1:1 也不是 7:1:2。

三套划分函数同时存在，调用方靠"哪里调到了"决定行为。

**建议**：合并为一个 `split_few_shot(dataset, shot_num, train_ratio=..., seed=...)`；明确语义并写在文档。

### 2.9 Prompt/Pretrain/Tasker 之间的循环依赖风险

- `pretrain/MultiGPrompt.py:4` 从 `prompt_graph.prompt` import `DGI/GraphCL/Lp/...`。
- `prompt/MultiGprompt.py` 同时定义了同名类 `DGI`、`GraphCL`、`Lp`。
- `pretrain/__init__.py` 又 `from .GraphCL import GraphCL`、`from .DGI import DGI`。

结果：`prompt_graph.prompt.DGI` 和 `prompt_graph.pretrain.DGI` 是两个完全不同的类，但都对外可见。日后 import 错一个就是非常隐晦的 bug。

**建议**：重命名 `prompt/MultiGprompt.py` 内的类（如 `MultiGpromptDGI`、`MultiGpromptLp`），消除歧义；或者把多任务版本放到 `prompt_graph.prompt.multigprompt` 子模块下，加上明确前缀。

---

## 3. 代码质量问题

### 3.1 命名混乱

| 现象 | 位置 | 建议 |
|---|---|---|
| `Train`（首字母大写） vs `train` | `graph_task.py:65` vs `node_task.py:92` | 统一小写 |
| `pg_opi`、`answer_opi`、`optimizer` 并存 | `task.py:57-70` | 统一为 `prompt_optimizer`、`head_optimizer` 等可读名 |
| `weigth_init` | `GPPTPrompt.py:52`、`node_task.py:242` | typo，应为 `weight_init` |
| `out_dim` vs `output_dim` | 全包 | 统一为 `output_dim` |
| `num_layer` vs `num_layers` | `get_args.py:63` 和 `97` 都定义但语义不同 | 这是真 Bug，CLI 上 `--num_layer` 和 `--num_layers` 是两个独立参数 |
| `gln` 参数 | `pretrain/base.py:7` | 应是 `num_layers`，单字母变量难懂 |
| `nb_nodes`、`nb_graphs`、`nb_classes` | `utils/process.py` | TF 风格，与项目其它地方不一致 |

### 3.2 拼写错误

仓库里随处可见同一个错别字 `eopch`（应为 `epoch`），在 `tasker/node_task.py`、`tasker/graph_task.py`、`pretrain/GraphCL.py`、`pretrain/DGI.py` 等都出现，且都在打印的 early-stopping 日志里。还有 `Bengin to evaluate`、`prepare induce graph`、`negetive_sample`（应为 `negative_sample`，在 `MultiGPrompt.py:38` 出现）等。

**建议**：一次性 grep 修一遍，避免后续每次看 log 都尴尬。

### 3.3 缩进风格混杂

- `node_task.py` 全文使用 **6 空格** 缩进（不是 4 也不是 tab）。`graph_task.py` 是标准 4 空格。
- `utils/seed.py` 第 38-49 行用 **tab**，其他函数用空格。
- 多个文件混用 CRLF / LF。

**建议**：引入 `ruff format`（或 `black`） + `pre-commit`，CI 上强制；这次一次性格式化全包，后续靠 hook 保持。

### 3.4 大量重复 import / 死 import

每个 model 文件开头都有：
```python
import sklearn.linear_model as lm   # 完全没用
import sklearn.metrics as skm        # 完全没用
import sklearn.linear_model as lm    # 第二次！
import torch, gc                     # gc 没用
import torch as th                   # 同一个文件里又写 import torch
```
说明这些文件是从某份模板 copy-paste 的，没人 review 过 import。

**建议**：跑 `ruff --select F401,F811` 自动清理；做一次顶层 import 审计。

### 3.5 大量 commented-out 代码

- `node_task.py:351-425`：~75 行被注释掉的 MultiGprompt 旧实现，包括路径、训练循环，完全是噪音。
- `tasker/graph_task.py:218-233, 274-284, 377-391, 422-433`：成段的 commented-out 备选 GPPT 实现。
- `pretrain/DGI.py:58-105`：旧 `pretrain_one_epoch` 注释保留。
- `load4data.py:64-72, 157-176`：成段类别划分备选代码。
- `prompt/AllInOnePrompt.py:42` 上方有死注释。

**建议**：用 git 来保留历史，仓库里删掉。如果某段是真的"暂存方案"，加 `TODO(name, issue#)` 而不是注释代码。

### 3.6 165+ 处 `print()`

整个包没有 logging，所有 epoch loss、early stop 信息都是 `print`，没有级别、没有格式化、没法静音。在 bench/扫超参时，stdout 噪音爆炸。

**建议**：
```python
# prompt_graph/utils/logging.py
import logging
def get_logger(name: str) -> logging.Logger: ...
```
每个模块用 `logger = get_logger(__name__)`，CLI 上加 `--log-level`。原 print 可以批量替换。

### 3.7 没有任何单元测试

```bash
$ find . -name 'test_*' -o -name 'tests'
# 空
```
项目体量已经超过几千行，零测试意味着任何重构都是赌博。

**建议**：
- 第一波：给 `load4node`、`load4graph`、`node_sample_and_save` 这种纯函数加 smoke test（Cora/MUTAG 跑通即可）。
- 第二波：给 `GNN` 工厂、`build_prompt`、`build_strategy` 加构造测试，确保所有支持的组合都能实例化、`forward` 一遍不报错。
- 第三波：给 `bench.py` 一个 mini-grid 的回归测试（固定 seed，对比 metric 不退化）。

### 3.8 缺少类型注解

包内几乎没有 type hints（`report_data.py` 是个例外）。在 Tasker/Strategy 这种多态结构里没注解就很难看清"`prompt` 到底是什么类型、有没有 `add` 方法"。

**建议**：从 `BaseTask.__init__` 签名开始加，逐步推进到 `model/`、`prompt/`、`evaluation/`。配 `mypy --strict prompt_graph.utils` 作为最起码的保护。

### 3.9 `GraphTask.run()` 双路径分裂

`graph_task.py:184` 起的 `if self.shot_num > 0` 和行 357 起的 `else` 两个分支各有 200+ 行训练 + 评估循环，**几乎一模一样**，只是数据切分方式不同。复制粘贴的差异已经在 `answer_epoch=50` 和 `answer_epoch=5` 上体现出来——说明两个分支早就发散了。

**建议**：抽出 `_train_one_split(train_loader, test_loader, ...)`，外层只负责生成数据切分。

### 3.10 隐式行为：`prompt_type='None'` 当成无 prompt

`task.py:44-55`：`'None'` 字符串既表示无 prompt 也表示无 pretrain。`return_pre_train_type` 把 `'None'` 也放进了 names 列表。这种"字符串当 enum"的模式在多个地方都出现：

```python
if self.pre_train_model_path == 'None':
if self.prompt_type == 'None':
if args.pretrain_task == 'None':
```

容易和真正的 `None` 弄混。

**建议**：CLI 入参允许 `None/null`，内部转成 Python `None`；或者定义 `class PromptType(StrEnum)`。

---

## 4. 工程实践问题

### 4.1 没有 `requirements.txt` / `pyproject.toml`

仓库根目录没有任何依赖声明。README 只在 badge 里写"PyTorch v1.13.1, python>=3.9"。`ogb`、`torch_geometric`、`torch_cluster`、`scikit-learn`、`torchmetrics`、`pandas`、`networkx`、`deprecated` 等都隐式依赖。

**建议**：补 `pyproject.toml`（推荐）或 `requirements.txt`，并锁定主版本号。考虑 `extras`：`prog[dev]`（test、ruff、mypy）、`prog[ogb]`（OGB 相关）。

### 4.2 `Node.zip`（661KB）、`ProG_pipeline.jpg`（531KB）、`Logo.jpg`（42KB）直接进 git

二进制资源放主仓库会迅速膨胀 `.git`。

**建议**：迁到 Releases 附件或 Git LFS；README 中改成下载链接。

### 4.3 `.gitignore` 把 `/Docs` 也忽略了

`.gitignore:9`：
```
/Docs
```
这意味着本文档 **默认不会被 git 跟踪**。再加上 `/Experiment`、`/data`、`/dataset` 也被忽略，团队成员之间无法共享文档和实验记录。

**建议**：从 `.gitignore` 移除 `/Docs`（明确这是项目文档）；如果不想被忽略的子目录在 `/Experiment`、`/data` 下，添加 `!` 例外规则。

### 4.4 隐式的 CWD 依赖

所有路径都是 `./Experiment/...`、`./data/...`，必须在仓库根目录运行才能工作。从 IDE 或外部脚本启动会找不到文件。

**建议**：用 `pathlib.Path(__file__).resolve().parents[N]` 推断仓库根目录，或者直接接受 `--data-root` / `--experiment-root` CLI 参数。

### 4.5 没有 CI / Lint / 任何质量门槛

没有 `.github/workflows/`、没有 `pre-commit-config`、没有 `ruff.toml`/`pyproject.toml [tool.ruff]`。

**建议**：最小 CI 三步：
1. `ruff check` + `ruff format --check`
2. import 一遍所有 `prompt_graph.*` 子模块（捕获 import-time bug）
3. `pytest -k smoke`（先只跑 1-2 个最快的）

### 4.6 配置散落

超参网格写死在 `bench.py`，模型默认值在 `model/*.py`，CLI 默认在 `get_args.py`，路径在 `load4data.py`，目录约定在 `tasker/*.py`。一旦想给某个数据集换 batch_size，需要改 3 处。

**建议**：把"运行配置"独立成 YAML：
```yaml
# configs/node/cora_graphcl_allinone.yaml
dataset: Cora
pretrain: GraphCL
prompt: All-in-one
search:
  lr: loguniform(1e-3, 1e-1)
  weight_decay: loguniform(1e-6, 1e-5)
  batch_size: [32, 64, 128]
trial_count: 10
```
`bench.py` 只是 `python -m prog.bench --config configs/node/cora_graphcl_allinone.yaml`。

### 4.7 `create_excel_for_bench.py` 是个一次性脚本，但被放在仓库根

它的作用是为所有数据集 × shot 组合预创建空白 Excel 表格，但写法是裸 for 循环，逻辑跟 `bench.py` 不对称（`bench.py` 只关心一个 dataset/prompt）。

**建议**：迁移到 `scripts/bootstrap_excel.py` 之类的位置，或者干脆并入 `bench.py`，第一次运行时按需创建表格。

---

## 5. 文档与仓库结构

### 5.1 `Docs/README.md` 与代码不同步

现有文档里：
- 列出 `LinkTask` 作为支持的任务类型（实际只支持 Cora 链路预测，未完成，见 §1.10）。
- 没有提到 `SUPT`、`MultiGprompt` 这些已经在 `prompt/__init__.py` 导出的 Prompt。
- "使用示例" 只列了 `python pre_train.py --help`，没说怎么跑 `bench.py`、没说 `PROG_USE_MPS` 这个隐藏环境变量。

**建议**：把 README 拆成：
- `Docs/architecture.md`（模块边界、Strategy 协议、registry 用法）
- `Docs/datasets.md`（每个数据集的来源、预期目录、特殊处理）
- `Docs/running.md`（pre_train / downstream_task / bench 三个入口的差异和示例）
- `Docs/improvements.md`（即本文档，跟踪改进计划）

### 5.2 `Tutorial/` 与 `Experiment/` 是两套并行的"示例"

`Tutorial/node_edge.py`、`Tutorial/node_graph.py` 是 Python 脚本而不是 notebook，跟其他 `*.ipynb` 教程的形态不一致；`Experiment/` 是实际跑出来的产物目录。新人很容易混淆。

**建议**：把 `Tutorial/*.py` 改成 notebook 或者并入 `examples/`；`Experiment/` 改名为 `runs/` 或 `outputs/`，更符合习惯。

### 5.3 顶层脚本和包内入口职责重叠

- `bench.py`：超参搜索 + 单次 NodeTask/GraphTask + 写 Excel
- `downstream_task.py`：单次 NodeTask/GraphTask，参数完全独立
- `pre_train.py`：预训练入口
- `create_excel_for_bench.py`：bench 的辅助初始化

四个脚本相互独立、各有一份 argparse、各有一份 device 推断。

**建议**：合并为 `python -m prog <pretrain|downstream|bench>`，共享 `get_args` 和 `get_device`。`create_excel` 收编为 `prog bench init`。

---

## 6. 建议的落地路线

下面给一个**不"推倒重来"也能逐步推进**的路线，每一步独立可合并。整体原则是：先建立"安全网"（baseline + 关键 Bug 修复），再做不破坏行为的局部抽象，最后才做能改变模块边界的大重构。

### 6.0 总览

| Phase | 目标 | 工期 | 前置 | 风险 | 主要产出 |
|---|---|---|---|---|---|
| Phase 0 | 建立 baseline 与协作约定 | 0.5 天 | — | 低 | `scripts/baseline.sh`、metrics 快照、改进追踪 issue |
| Phase 1 | 止血关键 Bug，正确性优先 | 0.5–1 天 | Phase 0 | 中（影响实验结果） | 8 个独立小 PR |
| Phase 2 | 清理（格式化 / 死码 / typo） | 1–2 天 | Phase 1 | 低 | 工具链就位 + 全包格式化 |
| Phase 3 | 局部抽象（不改变接口） | 3–5 天 | Phase 1、2 | 中 | `utils/device.py`、`utils/paths.py`、`model/gnn.py` registry |
| Phase 4 | 重构 Tasker（Strategy 化） | 1–2 周 | Phase 3 + smoke test | 高 | `PromptStrategy` 协议 + 6 个策略类，`run()` 收敛 |
| Phase 5 | 工程基建（CI/配置/日志） | 持续，并行展开 | Phase 0 | 低–中 | `pyproject.toml`、CI、YAML config、logging |
| Phase 6 | 文档对齐 | 1–2 天 | 各 Phase 收尾后 | 低 | 拆分后的 `Docs/` 体系 |

#### 6.0.1 实际落地进度（截至 2026-05-12）

| Phase | 状态 | 落地 commit / 备注 |
|---|---|---|
| Phase 0 | done | `e276670`（基线脚本 + baseline_metrics.md） |
| Phase 1 | done | `e276670`（8 条关键 bug 修复打成一个 squash） |
| Phase 2 | done | `87b01c2` 合并：Unit 1（工具链）+ Unit 2/3（死码+typo）+ Unit 4（reindent）+ Unit 5（GNN 死 import） |
| Phase 3 | done | `87b01c2` 合并：Unit 6（device 统一）+ Unit 7（paths）+ Unit 8（OGB RLock）+ Unit 9（GNN registry） |
| Phase 4 | done | 2026-05-12 落地：PRs #1-#8（Unit 14 foundation + Unit 15-20 六个 strategy + Unit 21 GraphTask 双分支合并）+ fix PR #17（MultiGprompt 隐藏 bug）。逐 prompt 渐进，每个 strategy 都通过 epoch1-5 loss ≤1e-3 校验。 |
| Phase 5.1 | done | `bbd1278`（pyproject + ruff + pre-commit） |
| Phase 5.2 | done | `ac9a1da`（tests/test_{data_loaders,factory,bench_smoke}.py） |
| Phase 5.3 | done | `bb98a4d`（GitHub Actions CI：lint + import smoke + pytest） |
| Phase 5.4 | done | `49b8088` 引入 `utils/get_logger`；2026-05-12 完成全包迁移：PRs #9-#14（Unit 22 tasker / Unit 23 pretrain / Unit 24 data+utils+prompt / Unit 25 evaluation / Unit 26 顶层脚本 + `--log-level`/`--quiet` CLI / Unit 27 Tutorial）。RESULT 行（` Final best ` / `Final Test:`）保留为 `print`。 |
| Phase 5.5 | done | 2026-05-12 落地：PR #15（Unit 28 YAML config + `configs/{cora_gpf,mutag_allinone,pubmed_gprompt}.yaml` + `pyproject.toml` 加 `pyyaml`，CLI 覆盖 YAML 覆盖默认值）。 |
| Phase 6.1 | done | `e0d9d4c`（.gitignore 移除 /Docs，IMPROVEMENTS/baseline_metrics/README 入库） |
| Phase 6.2-6.4 | done | 2026-05-12 落地：PR #19（拆分 `Docs/README.md` → `architecture.md` / `datasets.md` / `running.md`；新增 `CLAUDE.md`、`CONTRIBUTING.md`；顶层 `README.md` 移除"LinkTask"过时声明 + 加 Quickstart/Architecture 链接）。 |

**附带的 pre-existing bug 修复**（由 Phase 5.2 smoke tests 暴露，commit `b590fd2`）：

- `tasker/node_task.py:284` `range(1, self.epochs)` → `range(1, self.epochs + 1)`
- `tasker/{node,graph}_task.py` `int(self.epochs/self.answer_epoch)` → `max(1, int(...))`（3 处）

**端到端验证记录**（在 `87b01c2` 上跑）：

- `pytest tests/ -v` — 9 passed
- `python downstream_task.py --dataset_name Cora --gnn_type GCN --prompt_type GPF --downstream_task NodeTask --epochs 2 --shot_num 5 --seed 42 --device 0`（MPS）— PASS
- `python downstream_task.py --dataset_name MUTAG --gnn_type GCN --prompt_type All-in-one --downstream_task GraphTask --epochs 2 --shot_num 5 --seed 42 --device cpu` — PASS
- 同上 `--device 0`（MPS）— FAIL：`scatter_add_` on MPS 的 placeholder 限制，**非本批次引入**

#### 6.0.2 2026-05-12 之后的增量进度（截至 2026-05-22）

| 类别 | 内容 | commit / PR |
|---|---|---|
| Bug fix | `pre_train.py:GraphMultiGprompt` 已接通 `load4graph(pretrained=True)` + `GraphPrePrompt`，解决 §1.2 | `647d6c4` |
| Bug fix | Tutorial `node_graph.py` / `node_edge.py` 的 `edge_index` 未定义 | `ae04566` |
| Bug fix | `tasker` 移除 `task_num` 硬编码外层循环 + 统一 sample 路径，解决 §1.4 | `19f071f` |
| Refactor | `induced_graph.py` 抽出 `core builder` + 统一 `load_induced_graphs` 入口（§2.x） | `556e1e7` + `d35df5e` 配 smoke test |
| Refactor | `AllInOnePrompt.forward` 去掉 `to_data_list` 往返；contrast temperature 校验 | `cfbbdfb` |
| Feature | `FrontAndHead` 支持 multi_class / binary / regression | `ecc88b4` |
| Feature | `GPPT` k-means 支持 cosine / manhattan 距离 | `f9c2b61` |
| API cleanup | `LinkTask` 从公共 API 移除（§1.10）；`--num_iter` CLI 暴露 | `e76e20f` |
| New prompts | 9 个新 `prompt_type` 进入 `STRATEGY_REGISTRY`：`Prodigy` (`393cb08`) / `UniPrompt` (`7846e58`) / `SelfPro` (`a61b427`) / `ProNoG` (`4e63684`) / `DAGPrompT` (`92256f4`) / `PSP` + `RELIEF` + `GraphPrompter` MVP (`5178859`) / `RELIEF` full (`30401ca`) / `GraphPrompter` full (`a2abd91`) / `EdgePrompt` + `EdgePromptplus`（`strategies/edge_prompt.py`） | 见左 |
| Docs | baseline_logs gitignored | `5b561d5` |

---

**通用约定**（所有 Phase 都要遵守）：
- 每个 PR 一个目标，方便回滚。Phase 1 每条 Bug 一个 PR；Phase 3、4 每个抽象一个 PR。
- 旧 import 路径必须保留 alias，禁止"破坏式重命名"。重命名走两步：先加新名 + `DeprecationWarning`，下个版本再删旧名。
- Phase 4 之前，**任何 PR** 都不能改变已有 `bench.py` 命令的输出 metric（误差 ≤ 1e-4 视为不变）；Phase 4 期间通过 smoke test 守护。
- 分支命名：`fix/<bug-id>`、`refactor/<phase>-<topic>`、`chore/<topic>`。
- 提交信息前缀：`fix:`、`refactor:`、`chore:`、`docs:`、`test:`、`ci:`。

---

### Phase 0 — 准备（0.5 天）

**目标**：在动任何代码之前固化"当前是什么样"，给后续重构提供回归基准。

**前置**：无。

**步骤**：

1. **冻结一组金标准命令**。建议至少 3 组覆盖三种数据规模：
   ```bash
   # scripts/baseline.sh（新增）
   python bench.py --pretrain_task NodeTask  --dataset_name Cora     --prompt_type GPF        --gnn_type GCN --shot_num 5 --seed 42
   python bench.py --pretrain_task GraphTask --dataset_name MUTAG    --prompt_type All-in-one --gnn_type GCN --shot_num 5 --seed 42
   python bench.py --pretrain_task NodeTask  --dataset_name PubMed   --prompt_type Gprompt    --gnn_type GIN --shot_num 1 --seed 42
   ```
2. **跑通并记录 metric 快照**到 `Docs/baseline_metrics.md`（也可以是 JSON），格式：`dataset / prompt / gnn / shot / acc / f1 / auroc / 总耗时`。后续每个 Phase 合并前都要 diff 一次。
3. **建立改进追踪 issue / 看板**：把本文档 §1–§5 的条目逐条登记，标 `phase-N` 标签。
4. **拉协作分支**：从 `dev` 拉一个长期 `refactor/main` 分支，所有 Phase 分支基于它，最后再一次性合回 `dev`。

**Definition of Done**：
- `bash scripts/baseline.sh` 能在干净 checkout 上跑完（即使有 §1 的 Bug，跑得过的几路即可）；
- baseline metric 入库；
- issue 看板里至少有本文档列的全部条目。

**风险**：几乎为零。最容易踩的坑是"baseline 自己就受 §1 Bug 影响"——所以 Phase 1 修完后要再跑一次 baseline，并标注哪些 metric 因为修 Bug 而**预期变化**（如 §1.1、§1.5）。

---

### Phase 1 — 止血关键 Bug（0.5–1 天）

**目标**：让"跑一次 bench"得到的数字是可信的、不会意外 crash。

**前置**：Phase 0 baseline 已记录。

**修复清单**（每条独立 PR，可并行）：

| # | Bug | 文件 | 修法 | 是否影响指标 |
|---|---|---|---|---|
| 1 | GpromptTrain 提前 return | `prompt_graph/tasker/graph_task.py:121-152` | `return` 退一级缩进 | **是**（图任务 + Gprompt 全部需要重跑） |
| 2 | `pre_train.py` NameError | `pre_train.py:33-37` | 恢复 `load4graph` 调用，或在分支顶部 `raise NotImplementedError` | 否（之前根本跑不通） |
| 3 | `return_pre_train_type` 缺 fallback | `prompt_graph/tasker/task.py:141-145` | `os.path.basename` 取首段 + exact match + 匹配失败抛错 | 否（但 Excel 列名会变规范） |
| 4 | `split_induced_graphs` 测试集泄漏 | `prompt_graph/data/induced_graph.py:80` | 入参增加 `train_mask`，候选过滤非训练节点；并清除 `Experiment/induced_graph/*` 缓存 | **是**（few-shot 节点任务指标会下降，属于"修对了"） |
| 5 | `np.bool` 已被移除 | `prompt_graph/utils/process.py:131` | 改 `dtype=bool` | 否 |
| 6 | `seed_torch` 死代码 + tab 缩进 | `prompt_graph/utils/seed.py:38-49` | 删除 `seed_torch`；`__init__` 不导出 | 否 |
| 7 | `bench.py` 超参网格退化 | `bench.py:70-75` | 把 `np.linspace(512,512,200)` 改成 `[512]`、`np.linspace(-3,-1,1)` 改成 `[1e-2]`；显式 `num_iter=1` | 否（消除假象，行为不变） |
| 8 | `bench.py` device 不一致 | `bench.py:16-21, 109, 128, 135` | 临时方案：把 `runtime_device` 也透传给 `NodeTask/GraphTask`；最终方案在 Phase 3 完成 | 否 |

**Definition of Done**：
- 上述 8 个 fix 全部合并；
- `bash scripts/baseline.sh` 重新跑一遍，把"修 Bug 导致 metric 变化"的差异写进 `Docs/baseline_metrics.md` 的"Phase 1 之后"列；
- 修复 #4 后，所有 `Experiment/induced_graph/*.pkl` 缓存被删除并重新生成（旧缓存仍带泄漏数据）。

**风险**：
- **#1 和 #4 会让历史指标变差**——属于"修对了"的副作用，要在 PR 描述里明确说明，并通知正在用旧数据写论文的同事。
- **#3 修完后可能在某些 OGB 路径下抛异常**——这反而暴露了 `pre_train_model_path` 命名约定不严格的问题，按"匹配失败抛错"原则处理，让上游显式补名字。
- 若团队对历史 metric 有依赖，留一个 `--legacy-induced-graph` 临时开关一周作为过渡。

---

### Phase 2 — 清理（1–2 天）

**目标**：清掉视觉噪音，安装质量工具链，为后续大动作扫清"无关 diff"。

**前置**：Phase 1 合并。

**步骤**：

1. **工具链落地**（独立 PR，仅添加配置）：
   - 新增 `pyproject.toml` 或 `ruff.toml`，启用 `E,W,F,I,UP` 规则，line-length=100；
   - 新增 `.pre-commit-config.yaml`：`ruff check --fix`、`ruff format`、`trailing-whitespace`、`end-of-file-fixer`；
   - README 增加 "Setup pre-commit" 段落。
2. **死代码 / commented-out 代码批量清理**（独立 PR）：
   - `node_task.py:351-425`（旧 MultiGprompt 实现）；
   - `graph_task.py:218-233, 274-284, 377-391, 422-433`（备选 GPPT 实现）；
   - `pretrain/DGI.py:58-105`、`load4data.py:64-72, 157-176`、`prompt/AllInOnePrompt.py:42`。
3. **死 import**（独立 PR，`ruff --fix` 完成 90%）：
   - 6 个 GNN 文件里的 `import sklearn.linear_model`、重复 `import torch`、`import gc`；
   - 全包 `F401`、`F811` 一次清。
4. **缩进 + 编码统一**（独立 PR）：
   - `node_task.py` 全文从 6-space 转回 4-space；
   - 把 CRLF 统一为 LF；
   - `.editorconfig` 一并加入。
5. **拼写错误批量替换**（独立 PR）：
   - `eopch` → `epoch`、`weigth_init` → `weight_init`、`negetive_sample` → `negative_sample`、`Bengin to evaluate` → `Begin to evaluate`；
   - `weigth_init` 在 `GPPTPrompt.py:52` 和 `node_task.py:242` 都要改，作为函数名要同步重命名调用方。

**Definition of Done**：
- `ruff check` 和 `ruff format --check` 均通过；
- `pre-commit run --all-files` 通过；
- `git grep -E 'eopch|weigth_init|negetive_sample|Bengin'` 输出为空；
- `bash scripts/baseline.sh` metric 与 Phase 1 完全一致（误差 ≤ 1e-4）。

**风险**：
- 大规模格式化会让历史 `git blame` 失真。**对策**：把"纯格式化"提交放进 `.git-blame-ignore-revs`，并在仓库设置里启用。
- 拼写更名涉及函数签名（如 `weigth_init`）的，要 grep 调用方一并改；ruff 不会帮你。

---

### Phase 3 — 局部抽象（3–5 天）

**目标**：把散落的"基础设施代码"（device 选择、路径管理、GNN 构造）收拢到 utils 层，**不改变任何对外行为**。这一步是 Phase 4 大重构的前提。

**前置**：Phase 2 合并，工具链生效；至少 2 条 smoke test（见 Phase 5 §2）已落地。

**步骤**：

1. **`prompt_graph/utils/device.py`**（独立 PR）：
   ```python
   def get_device(spec: int | str | None = None) -> torch.device: ...
   ```
   - `int` → `cuda:N`（不可用降级 cpu + `logger.warning`）；
   - `'auto'` → CUDA > MPS > CPU；
   - `'mps'/'cpu'/'cuda:N'` → 透传；
   - `get_args.py` 增加 `--device {auto,cuda:N,mps,cpu}`，老的 `--device 0` 兼容；
   - `bench.py`、`task.py`、`pretrain/base.py`、`pretrain/MultiGPrompt.py` 全部改用 `get_device`；
   - **删除** `PROG_USE_MPS` 隐式环境变量，README 中说明替代为 `--device mps`。
2. **`prompt_graph/utils/paths.py`**（独立 PR）：
   - 暴露 `DATA_ROOT`、`EXPERIMENT_ROOT`、`induced_graph_dir(dataset)`、`sample_dir(task, shot, dataset)`；
   - 默认走 `Path(__file__).resolve().parents[2]`，可被 `PROG_DATA_ROOT` / `PROG_EXPERIMENT_ROOT` 覆盖；
   - 替换 `load4data.py` 内 `'data/TUDataset'`、`'./dataset'`、`node_task.py` 内 `'./Experiment/sample_data/...'`、`bench.py` 内 `'./Experiment/induced_graph/...'` 等所有裸字符串路径。
3. **`prompt_graph/model/gnn.py` + registry**（独立 PR，合并 6 个 GNN 文件）：
   - 一个 `class GNN(nn.Module)`，构造参数 `conv_type, num_layer, in_dim, hid_dim, ...`；
   - `_GNN_REGISTRY = {'GCN': GCNConv, 'GAT': GATConv, ...}`，GIN 用闭包工厂；
   - `model/__init__.py` 继续暴露 `from .gnn import GNN as GCN` 等别名；
   - 旧 `model/GCN.py` ... 6 个文件**保留为薄 re-export shim**，附 `DeprecationWarning`，下版本再删；
   - `BaseTask.initialize_gnn` 和 `pretrain/base.py` 都改用 `from prompt_graph.model import build_gnn`。
4. **`ogb_torch_load_compat` 改造**（独立 PR）：
   - 不再 monkey-patch 全局 `torch.load`；
   - 改成只在 OGB dataset 构造时局部包一层，或 fork OGB 加载逻辑传 `weights_only=False`；
   - 若改造代价高，至少加 `threading.RLock` 保护并加单元测试覆盖异常路径。

**Definition of Done**：
- `grep -rE "torch\.cuda\.is_available\(\)" prompt_graph/ bench.py pre_train.py downstream_task.py` 只在 `utils/device.py` 中出现；
- `grep -rE "'./Experiment|'./dataset|'data/TUDataset'" prompt_graph/ bench.py` 输出为空；
- `prompt_graph/model/` 下业务逻辑只在 `gnn.py`，其它 6 个文件是 ≤ 5 行的 shim；
- baseline metric 与 Phase 2 一致；
- smoke test 全部通过。

**风险**：
- GNN 合并最容易出岔：6 个文件虽然"看起来"差不多，但其中可能有人改过 `forward` 里 `JK` 或 `pool` 的一处而没同步——**对策**：合并前先 `diff` 两两比对，把任何"看起来一致"的差异先记录到 issue。
- 路径替换会改变缓存目录的实际位置——**对策**：在 README 显式注明并提供 `PROG_EXPERIMENT_ROOT=$(pwd)/Experiment` 的迁移命令。

---

### Phase 4 — 重构 Tasker（1–2 周）

**目标**：把 `NodeTask` / `GraphTask` 里"按 prompt_type 大 if/elif 分发"的逻辑收敛到独立的 Strategy 类，`run()` 收敛到 50 行以内。**没有 Phase 5 §2 的 smoke test 兜底不要开**。

**前置**：Phase 3 合并 + smoke test 至少覆盖每个 prompt_type × 一个数据集 × 一次 epoch。

**步骤**（分 PR，按 prompt_type 渐进式迁移）：

1. **协议设计 PR（仅新增，不替换）**：
   ```python
   # prompt_graph/tasker/strategy.py
   class PromptStrategy(Protocol):
       name: ClassVar[str]
       def setup(self, ctx: TaskContext) -> None: ...
       def configure_optimizer(self, ctx: TaskContext) -> torch.optim.Optimizer: ...
       def train_epoch(self, ctx: TaskContext, loader_or_data) -> float: ...
       def evaluate(self, ctx: TaskContext, loader_or_data) -> Metrics: ...
   ```
   - `TaskContext` 是 dataclass，封装 `gnn`、`prompt`、`answering`、`device`、`hidden_dim` 等子模型；
   - 增加 `STRATEGY_REGISTRY`，但**不**让 `BaseTask` 使用它，现有逻辑保持原状。
2. **试点：`NoneStrategy`（最简单的标准微调）**：
   - 把 `prompt_type == 'None'` 的训练/评估路径搬到 `strategies/none.py`；
   - `BaseTask.run` 在 `prompt_type == 'None'` 时优先走 strategy，其它仍走老代码；
   - smoke test 跑通后合并。
3. **依次迁移其它 prompt_type**（每个一个 PR）：
   - 顺序建议：`GPF` → `GPF-plus` → `Gprompt` → `All-in-one` → `GPPT` → `MultiGprompt`（最复杂留最后）；
   - 每完成一个，删掉对应的 if/elif 分支；
   - 每个 PR 后都跑一遍 smoke test + baseline。
4. **GraphTask.run 双分支合并**（独立 PR）：
   - 抽 `_train_one_split(train_loader, test_loader, ...)` 复用；
   - 外层只负责生成数据切分；
   - `answer_epoch` 不一致的差异作为参数显式传入（可能是 §1 级别的隐藏 Bug，PR 描述里需和团队对齐）。
5. **`BaseTask` 初始化顺序明确化**（独立 PR）：
   - 抽 `_build_dataset` / `_build_model` / `_build_optimizer` 钩子；
   - `NodeTask` / `GraphTask` 只 override 钩子，禁止再在 `run()` 里"延迟初始化"。

**Definition of Done**：
- `tasker/{node_task,graph_task}.py` 加起来不超过 600 行（当前合计 ~1500 行）；
- 新增任何 prompt_type 只需要写一个 `XxxStrategy` 类 + registry 注册；
- baseline metric 与 Phase 3 一致；
- smoke test 全部通过，且覆盖 prompt_type × {NodeTask, GraphTask}。

**风险**：
- **最大头**：每个 prompt 类型隐藏的细节差异（如 `Gprompt` 需要 `mean_centers`、`All-in-one` 有自己的 inner loss）容易在迁移中丢掉；**对策**：每次迁移前后 diff 单次训练的 loss 曲线，要求 epoch 1–5 的 loss 序列误差 ≤ 1e-3。
- **第二大头**：`MultiGprompt` 与 `pretrain/MultiGPrompt.py` / `prompt/MultiGprompt.py` 强耦合，且类名歧义（§2.9），建议放在最后并先做 §2.9 的重命名。
- 中途若发现某个 prompt_type 当前实现就是错的（如 §1.1），先停下来开 Phase 1 风格的 fix PR，不要混在 Strategy 迁移里。

---

### Phase 5 — 工程基建（持续，与 Phase 3、4 并行）

**目标**：补齐"现代 Python 项目"的标配。可分布到各 Phase 之间穿插推进，不必一次做完。

**步骤**（按可独立合并程度排序）：

1. **依赖声明 + 安装文档**（最早做，与 Phase 2 同期）：
   - 新增 `pyproject.toml`（推荐 PEP 621）；
   - 主依赖：`torch>=1.13`、`torch-geometric`、`torch-cluster`、`torch-scatter`、`torch-sparse`、`ogb`、`pandas`、`openpyxl`、`scikit-learn`、`torchmetrics`、`networkx`、`numpy`、`deprecated`、`tqdm`；
   - extras：`prog[dev]` 含 `pytest`、`ruff`、`pre-commit`、`mypy`；`prog[ogb]` 含 `ogb`、`outdated`；
   - 锁定主版本：`torch>=1.13,<3`、`numpy<2`（兼容老 OGB）；
   - README 增加"3 步装好"段落：`pip install -e .[dev,ogb]`。
2. **smoke test 第一波**（必须在 Phase 4 之前）：
   - `tests/test_data_loaders.py`：`load4node('Cora')`、`load4graph('MUTAG')` 不抛错、维度正确；
   - `tests/test_factory.py`：`build_gnn(name, ...)` 对每个 `name` 都能 forward 一个随机 batch；
   - `tests/test_bench_smoke.py`：固定 seed + 1 epoch + 1 trial 跑通 `do_config_bench`，断言 metric 在合理范围。
3. **CI**（独立 PR，紧跟 §1 之后）：
   - `.github/workflows/ci.yml`：`push` / `pull_request` 时跑 `ruff check`、`ruff format --check`、`pytest -q`；
   - matrix：`python-version: ['3.9', '3.11']`，CPU 跑；
   - 单 job 时间预算 5–8 分钟。
4. **logging 替换**（与 Phase 4 同期，可分模块）：
   - 新增 `prompt_graph/utils/logging.py`：`get_logger(name)` 返回带统一 formatter 的 logger；
   - 第一波：`tasker/*.py`、`pretrain/*.py` 内的 epoch / early-stop print；
   - 第二波：`bench.py`、`pre_train.py`、`downstream_task.py` 的入口 print；
   - CLI 增加 `--log-level` / `--quiet`。
5. **YAML 配置**（Phase 4 收尾时引入）：
   - 新增 `configs/` 目录，给 baseline.sh 的 3 组命令各写一份 yaml；
   - `bench.py` 增加 `--config` 参数，支持"yaml 覆盖 CLI 默认值，CLI 显式传入再覆盖 yaml"；
   - 旧 CLI 调用保持有效。

**Definition of Done**：
- 新成员只需 `git clone && pip install -e .[dev,ogb] && bash scripts/baseline.sh` 就能跑起来；
- CI 在 `main` 分支保持绿色；
- 任何 epoch print 都通过 logger，可用 `--quiet` 完全静音。

---

### Phase 6 — 文档对齐（1–2 天）

**目标**：让文档和代码状态一致，让 `Docs/` 本身进入版本控制。

**步骤**：

1. **`.gitignore` 调整**（独立 PR，最早可做）：
   - 移除 `/Docs`、保留 `/data` `/Experiment` `/dataset`；
   - 若 `/Docs` 下有不该跟踪的子目录（如 `Docs/private/`），用例外规则 `Docs/private/` 单独忽略；
   - 历史上若已存在被忽略的 `Docs/*.md`，需要 `git add -f` 一次性入库。
2. **拆分 `Docs/README.md`**（Phase 3、4 完成后做最准确）：
   - `Docs/architecture.md`：模块边界、`PromptStrategy` 协议、registry 用法、初始化顺序；
   - `Docs/datasets.md`：每个数据集的来源、根目录约定、特殊处理（如 ENZYMES 的节点分类语义）；
   - `Docs/running.md`：`pre_train` / `downstream_task` / `bench` 三个入口、`--config` 用法、`--device` 说明；
   - `Docs/IMPROVEMENTS.md`（本文档）：持续维护，每完成一个 Phase 在条目前打 ✅。
3. **CLAUDE.md / CONTRIBUTING.md**：
   - 把 Phase 0 的"baseline + 分支命名 + 提交前缀"沉淀进 CONTRIBUTING；
   - 把 §1.5 这类"如果你看到 'Test data may leak' 这种注释请按 issue tracker 处理"的项目级约定写进 CLAUDE.md。
4. **更新顶层 `README.md`**：
   - 移除"支持 LinkTask"的说法直到 §1.10 被真正修复；
   - 增加 `Quickstart`、`Architecture` 章节链接。

**Definition of Done**：
- 新加入项目的人只读 `Docs/` 就能理解模块边界与运行方式；
- `Docs/IMPROVEMENTS.md` 的条目状态与代码状态完全一致；
- 顶层 `README.md` 不再有"未实现 / 已删除"的过时信息。

---

### 6.x 进度跟踪建议

建议在仓库根放一个 `STATUS.md` 或 GitHub Project Board，把上面表格中的 Phase 当作 Milestone，每条 §1–§5 的条目当作 issue，状态扭转：
- `backlog` → `in-phase-N` → `in-review` → `done`；
- 每个 Phase 收尾打一个 git tag：`refactor-phase-1`、`refactor-phase-2`，方便后续 `git diff refactor-phase-1..refactor-phase-3 -- prompt_graph/` 复盘。

---

## 附录 A：本次扫描中**未深入**的部分

下面这些子模块本次只做了基本的目录浏览，没有逐行审；如果之后要继续审计，建议作为下一轮的范围：

- `prompt_graph/pretrain/{GraphMAE,SimGRACE,Edgepred_GPPT,Edgepred_Gprompt}.py` — 仅核对了入口签名。
- `prompt_graph/prompt/{MultiGprompt,SUPT,GPPTPrompt}.py` 的内部数学实现 — 仅扫了类签名和明显问题，没核对算法正确性。
- `prompt_graph/data/batch.py`、`pooling.py` — 未读。
- `Tutorial/downstream_task.ipynb` — 未读，可能也存在过期 API。

> 截至 2026-05-22，附录 A 列出的 4 类未深审项**仍未深审**；2026-05-12 之后新增的 9 个 strategy（`Prodigy` / `UniPrompt` / `SelfPro` / `ProNoG` / `DAGPrompT` / `PSP` / `RELIEF` / `GraphPrompter` / `EdgePrompt(+plus)`）也只有 smoke test 覆盖，未做算法层 review，建议下一轮一并审。

---

## 7. 当前未完成项清单（生成于 2026-05-22）

> 这一节是 §1-§5 + 附录 A + README TODO List + 源码 grep `TODO` 的一个**当前快照**。每条都有所属类别、优先级。每个 Phase 收尾或大版本发布时刷新本表。

**优先级口径**：

- **P0**：当前文档与代码强烈不一致，会直接误导新用户 / agent，必须立刻修。
- **P1**：架构/质量遗留，影响后续维护或回归守护，应该尽快排期。
- **P2**：长期 TODO，对正确性 / 用户体验有可衡量影响，下一个迭代周期内做。
- **P3**：锦上添花、研究类 follow-up、或仅在特定场景下踩到的边缘问题。

**类别速查**：A.README 残留 / B.重构遗留 / C.待深审 / D.源码 TODO / E.文档同步 / F.回归守护。

| P | 类别 | ID | 标题 | 来源 / 备注 |
|---|---|---|---|---|
| P0 | E.文档同步 | `docs-new-prompts-not-listed` | `Docs/{architecture,running}.md` + `.github/copilot-instructions.md` 仍只列 6-7 个 strategy，未涵盖 9 个新方法 | 本批次（2026-05-22）已修，`architecture.md` §3 / `running.md` §1 / `copilot-instructions.md` 同步 |
| P1 | E.文档同步 | `docs-improvements-status-sync` | IMPROVEMENTS §1.2 / §1.10 / 附录 A 状态需同步 | 本批次已修 |
| P1 | E.文档同步 | `docs-copilot-instructions-sync` | `.github/copilot-instructions.md` 沿用了 `CLAUDE.md` 的旧措辞（LinkTask / GraphMultiGprompt） | 本批次已修 |
| P1 | B.重构遗留 | `claude-initialize-prompt-optimizer` | `tasker/task.py:initialize_prompt` & `initialize_optimizer` 仍 if/elif 分发，未迁入 `PromptStrategy` —— Phase 4 收尾的 follow-up | `Docs/architecture.md` §4 备注 + `CLAUDE.md` §4.2 |
| P1 | B.重构遗留 | `graphmultigprompt-now-done` | 已 ✅（commit `647d6c4`），相关文档段落已同步 | 本批次完结 |
| P1 | F.回归守护 | `docs-baseline-metrics-empty` | `Docs/baseline_metrics.md` phase-1 ~ phase-6 列全空，每次 PR 应跑 `scripts/baseline.sh --tag <phase-X>` 回填 | 影响回归判定 |
| P2 | A.README 残留 | `readme-pretrain-infograph` | 新增 InfoGraph / ContextPred / AttrMasking / GraphLoG / JOAO 5-6 种预训练范式 | README §405-415 |
| P2 | A.README 残留 | `readme-induced-graph` | 改进 induced graph 生成算法 + 简化 3 类 generate-func | 已部分（`556e1e7` 抽 core builder），算法本身未改 |
| P2 | A.README 残留 | `readme-tutorial-notebook` | `Tutorial/` 脚本 → notebook + 数据处理 demo | README §405-415 |
| P2 | B.重构遗留 | `claude-enzymes-labelcols` | `load4node('ENZYMES')` 用最后 3 列做 one-hot label 的隐式约定要在代码里加注释 | `CLAUDE.md` §4.5 + `Docs/datasets.md` §5.1 |
| P2 | B.重构遗留 | `claude-mps-allinone-mutag` | MPS 上 `GraphTask + All-in-one + MUTAG` 的 `scatter_add_` `NotImplementedError`，需要替代实现或显式 dispatch | `CLAUDE.md` §4.6 |
| P2 | B.重构遗留 | `linktask-stale` | `link_task.py` 文件保留但已从公共 API 删除；若要复活按 §1.10 清单走 | 本节 §1.10 |
| P2 | D.源码 TODO | `code-edgepred-gprompt-todo` | `pretrain/Edgepred_Gprompt.py:100` — `GraphPrompt customized node embedding computation` 未实现 | grep TODO 唯一存活项 |
| P3 | A.README 残留 | `readme-comprehensive-doc` | 参照 PyG 写完整 usage doc | README §405-415 |
| P3 | A.README 残留 | `readme-deepgcn` | 支持 `DeepGCNLayer`（PyG `nn.models.DeepGCNLayer`） | README §405-415 |
| P3 | B.重构遗留 | `claude-allinone-return-asymmetry` | `AllInOneStrategy.train_epoch` NodeTask 返 `answer_loss`、GraphTask 返 `pg_loss` —— 改之前需先加 baseline 列 | `CLAUDE.md` §4.3 |
| P3 | B.重构遗留 | `claude-graphtask-tuple-shape` | `GraphTask.run` 返回 9-tuple / 4-tuple（few-shot vs full）—— 文档要求"不要为了优雅而统一" | `CLAUDE.md` §4.4 |
| P3 | C.待深审 | `audit-pretrain-internals` | 深审 `pretrain/{GraphMAE,SimGRACE,Edgepred_GPPT,Edgepred_Gprompt}` 算法正确性 | 附录 A |
| P3 | C.待深审 | `audit-prompt-math` | 深审 `prompt/{MultiGprompt,SUPT,GPPTPrompt}` 数学实现 | 附录 A |
| P3 | C.待深审 | `audit-data-batch-pooling` | 深审 `data/batch.py` + `pooling.py` | 附录 A |
| P3 | C.待深审 | `audit-tutorial-notebook` | `Tutorial/downstream_task.ipynb` 可能有过期 API | 附录 A |
| P3 | C.待深审 | `audit-new-strategies` | 9 个新 strategy（`Prodigy` / `UniPrompt` / `SelfPro` / `ProNoG` / `DAGPrompT` / `PSP` / `RELIEF` / `GraphPrompter` / `EdgePrompt(+plus)`）只有 smoke test 覆盖，未做算法层 review | 2026-05-12 之后增量，附录 A 延伸 |
| P1 | B.重构遗留 | `strategy-initialize-optimizer-missing-branches` | ✅ **已修 2026-05-22**：`tasker/task.py:initialize_optimizer` 在 `("PSP", "RELIEF", "GraphPrompter")` 分支里追加了 `Prodigy` / `EdgePrompt` / `EdgePromptplus` / `UniPrompt`。同时把 `node_task.py` / `graph_task.py` 的 `_*_ctx()` 里 `self.decay` 全部改成 `self.wd`（typo） | `tests/test_strategy_new_prompts.py` 8 个 XFAIL → PASS |
| P1 | B.重构遗留 | `strategy-uniprompt-init-missing-k` | ✅ **已修 2026-05-22**：`BaseTask.initialize_prompt` 用 `k=getattr(self, "uniprompt_k", 5)` 显式传 k | Node/Cora/UniPrompt XFAIL → PASS |
| P1 | B.重构遗留 | `strategy-dagprompt-graph-init-mismatch` | ✅ **已修 2026-05-22**：`TaskContext` 加 `param_center_embeddings` 字段；`DAGPrompTStrategy.train_epoch` 先用 `center_embedding_multihop` 算 empirical centers 再 apply learnable residual（之前 `param_center_embeddings(out, y)` 的两参调用对不上 `forward(centers)` 的单参签名） | Graph/MUTAG/DAGPrompT XFAIL → PASS |
| P1 | B.重构遗留 | `strategy-relief-args-missing-hid-dim` | ✅ **已修 2026-05-22**：RELIEF Args 类加 `hid_dim = None` 字段，`setup()` 在生效前从 `ctx.hid_dim` 注入 | RELIEF 第一层 init 通过，但 attach_prompt 内仍有 scatter 2-D 问题，详见下条 |
| P1 | B.重构遗留 | `strategy-edgeprompt-dim-list-mismatch` | ✅ **已修 2026-05-22**：`BaseTask.initialize_prompt` 把 `EdgePrompt` / `EdgePromptplus` 的 `dim_list` 从 `[hid_dim] * num_layer` 改成 `[input_dim] + [hid_dim] * (num_layer - 1)`，匹配 `model/gnn.py` "prompt applied before each conv" 的语义 | Graph/MUTAG/EdgePrompt + EdgePromptplus XFAIL → PASS |
| P1 | B.重构遗留 | `strategy-graphprompter-graph-readout` | ✅ **已修 2026-05-22 (Bugfix 第二批)**：`GraphPrompterModel.forward` 末尾 slice 出真实 supernode 的 logits 行（去掉 kNN-augmented prompt 行）；`GraphTask.run` 加 GraphPrompter eval 分支（之前漏写 → `UnboundLocalError`） | Graph/MUTAG/GraphPrompter XFAIL → PASS（acc 0.3987） |
| P1 | B.重构遗留 | `strategy-relief-scatter-shape` | ✅ **已修 2026-05-22 (Bugfix 第二批)**：根因是 `_node_to_state` 把 prompt 当作 `batch` positional arg 传给 GNN（→ 触发 pool 的 2-D scatter）；改成 `gnn(..., prompt=prompt)` kwarg。配套修了 `gnn.py` RELIEF 直接扰动路径（之前 `x + prompt` 在 hid_dim 上，且漏掉 input → conv 维度），改成 `h_list[0] + prompt`；`node_task.py:_relief_ctx` 注入 `svd_dim=input_dim`、`max_num_nodes=data.num_nodes`、`relief_max_attach_steps=50`（cap 否则 Cora 一个 epoch 2708 GNN forward，太慢） | RELIEF NodeTask Cora 跑通（slow opt-in），acc 0.1903，单 fold ~7 分钟 |
| P2 | B.重构遗留 | `strategy-relief-graphtask-no-init` | ✅ **已修 2026-05-22 (Batch-3 / P2.1)**：`task.py:initialize_prompt` 加 GraphTask 分支（用 `max(d.num_nodes for d in self.dataset)` 推 num_nodes），`graph_task.py` 加 `_relief_ctx()` + train/eval dispatch | Graph/MUTAG/RELIEF acc=0.4867 |
| P2 | B.重构遗留 | `strategy-relief-perf-cap` | ⚠️ **架构性重做尝试失败 (Batch-3 / P2.2)**：用 induced subgraph 路径替代单图路径，结果**慢且差**（13min/fold vs 7min/fold，acc 0.14 vs 0.19）。原因：(1) 2400+ eval subgraph × 174 nodes/sub → per-batch GNN forward 在 11k-node tensor 上耗时抵消了内层循环 cap 的收益；(2) NodeTask 下每个 subgraph.y 是 center node label，但 tasknet 读 graph_emb（whole-subgraph pool），loss signal 错位。**已回退**，保持单图 + `attach_prompt` step cap=50 路径。**副作用 ✅**：把 step cap 同样接到 `train_policy_epoch`（带 `effective_nodes_per_graph` clamp 保证 index 在 bound 内），改善 GraphTask 大图 RELIEF 的最坏情况 | `Node/Cora/RELIEF` 仍 SKIP（在 sweep 中标 functional but slow） |

**类别速查**：A.README 残留 / B.重构遗留 / C.待深审 / D.源码 TODO / E.文档同步 / F.回归守护。

### 7.1 Benchmark 覆盖现状（2026-05-22 末次更新）

测试 / 跑分双通道：

| 通道 | 入口 | 范围 | 跑一次的耗时 | 状态（2026-05-22 末次） |
|---|---|---|---|---|
| 1-epoch pytest smoke | `tests/test_strategy_new_prompts.py` + `tests/test_strategy_*.py` | 全 17 个 strategy × {Node Cora / Graph MUTAG} 的可用组合 | ~14 分钟（**54 passed + 1 skipped + 2 xfailed**） | CI 跑；xfail 守护下面 7.3 的 2 条 P1 |
| Phase-0 金标准 | `scripts/baseline.sh` | 3 个冻结 case：Cora+GPF / MUTAG+All-in-one / PubMed+Gprompt | ~30-60 分钟（200 epoch） | metric 漂移 > 1e-4 阻止合并；case 集合**不可扩** |
| 全 prompt 覆盖 sweep | `scripts/benchmark_all_prompts.sh` | 14 Node + 10 Graph + 3 SKIP（默认）/ 总 27 | `--fast` ~3.5 分钟（CPU, num_iter=1, 50 epoch） | 失败不 abort；不写 `baseline_metrics.md` |

> 通道 3 是新加的。它解决了"baseline.sh 只覆盖 3 个 prompt"的盲区，同时不影响 Phase-0 漂移守护。

### 7.2 末次 sweep 结果（`bash scripts/benchmark_all_prompts.sh --fast --tag bugfix3-final`，2026-05-22 第三批 bugfix / P2 round 后）

- **26 PASS / 0 FAIL / 2 SKIP**（log: `scripts/baseline_logs/bugfix3-final_20260523_000355_all_prompts.log`；条目级表：`scripts/baseline_logs/bugfix3-final_summary.md`）。
- 2 个 SKIP：
  - `Node/Cora/MultiGprompt` — 设计上需要 pretrained checkpoint，设 `MULTIGPROMPT_PRETRAIN_PATH` 后会自动跑。
  - `Node/Cora/RELIEF` — functional 但慢（~7 min/epoch on Cora）。架构性重做尝试失败（见 §7.3 P2.2），保留 step cap + 单图路径。
- pytest：**56 passed, 1 skipped, 1 deselected, 0 xfailed**（新加 Graph/MUTAG/RELIEF 参数化）。
- 关键数字（**仅 P2 影响以下条目**，其余 unchanged）：
  - `Graph/MUTAG/DAGPrompT`：F1 0.6198（之前 0.0000 占位）、AUROC 0.6270（之前 0.0000）。 ✅ P2.3 修复
  - `Graph/MUTAG/RELIEF`：**NEW** acc=0.4867 / F1=0.3195。✅ P2.1 修复
- 顶部三甲不变：NodeTask `Gprompt` 0.6328，GraphTask `All-in-one` 0.7747 (AUROC 0.8753)，`None` 0.7240。

### 7.3 当前仍 open 的 P1/P2 复盘（2026-05-22 末次）

#### P1 strategy bugs：7/7 全部 ✅

| ID | 状态 |
|---|---|
| `strategy-initialize-optimizer-missing-branches` | ✅ 已修 (batch-1) |
| `strategy-uniprompt-init-missing-k` | ✅ 已修 (batch-1) |
| `strategy-dagprompt-graph-init-mismatch` | ✅ 已修 (batch-1) |
| `strategy-relief-args-missing-hid-dim` | ✅ 已修 (batch-1) |
| `strategy-edgeprompt-dim-list-mismatch` | ✅ 已修 (batch-1) |
| `strategy-graphprompter-graph-readout` | ✅ 已修 (batch-2) |
| `strategy-relief-scatter-shape` | ✅ 已修 (batch-2) |

#### P2 strategy follow-ups：2/3 ✅，1 个 documented 为 design constraint

| ID | 状态 |
|---|---|
| `strategy-relief-graphtask-no-init` (P2.1) | ✅ 已修 (batch-3) — GraphTask 路径接通，Graph/MUTAG/RELIEF acc=0.4867 |
| `strategy-relief-perf-cap` (P2.2) | ⚠️ 架构性重做尝试失败 → 保留单图 + step cap=50；副作用是把 cap 接到了 `train_policy_epoch`（带 clamp）。`Node/Cora/RELIEF` 在大单图上是 design constraint：~7 min/fold 是 RL 算法基本开销。建议在大单图 NodeTask 上**用 Prodigy / GPPT / ProNoG / Gprompt 替代**（这些都秒级跑完） |
| `strategy-dagprompt-eval-placeholders` (P2.3) | ✅ 已修 (batch-3) — F1/AUROC/AUPRC 现在用 sklearn 计算，Graph/MUTAG/DAGPrompT 报出 F1=0.6198、AUROC=0.6270 |

**结论**：7 条 P1 + 2 条 P2 已修；剩 1 条 P2（RELIEF 单图 perf）登记为 design constraint，不阻塞功能。所有 26 个 (task × prompt) 组合在 sweep 里 PASS（除按设计 SKIP 的 2 个）。

---

*文档创建：2026-05-12  
扫描参考：`bench.py` `prompt_graph/` 全包 + 顶层脚本。  
末次状态更新：2026-05-22（bugfix sweep）。*
