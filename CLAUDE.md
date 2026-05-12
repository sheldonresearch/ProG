# CLAUDE.md

给 AI 协作者（Claude / Cursor / Copilot / 其他 LLM agent）的项目级约定。这里写的是**和"读 README + 看代码"得不到**的隐性约束、容易踩的坑、以及历史决策的"为什么"。

人类协作者请看 [`CONTRIBUTING.md`](./CONTRIBUTING.md) 和 [`Docs/architecture.md`](./Docs/architecture.md)。

---

## 1. 仓库速览

ProG 是一个 graph prompt learning 的实验框架。三个入口：`pre_train.py` / `downstream_task.py` / `bench.py`。核心包是 `prompt_graph/`。

如果你对仓库结构不熟，先读：

1. [`Docs/architecture.md`](./Docs/architecture.md) — 模块边界、`PromptStrategy` 协议、初始化顺序；
2. [`Docs/running.md`](./Docs/running.md) — CLI 标志与三个入口；
3. [`Docs/datasets.md`](./Docs/datasets.md) — 数据集与磁盘路径；
4. [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md) — 改进 roadmap，已落地 + 待办都在这里。

---

## 2. 常见任务的"权威入口"

| 你要做的事 | 不要从空白页猜，先看 |
|---|---|
| 新增一个 prompt_type | `prompt_graph/tasker/strategies/` 现有 6 个 strategy + `Docs/architecture.md` §3 |
| 新增一个 GNN backbone | `prompt_graph/model/__init__.py` 的 registry + `build_gnn` |
| 新增一个数据集 | `prompt_graph/data/load4data.py` + `prompt_graph/defines.py:NODE_TASKS/GRAPH_TASKS` |
| 新增 CLI 标志 | `prompt_graph/utils/get_args.py` 的 `_build_parser` + `get_args_by_call` + `DEFAULT_ARG_DICT` |
| 改路径约定 | `prompt_graph/utils/paths.py`（不要再硬编码 `./data` / `./Experiment`） |
| 打日志 | `from prompt_graph.utils import get_logger`，不要 `print` |

---

## 3. 反模式（请勿引入）

### 3.1 不要硬编码路径

```python
# BAD
torch.save(obj, "./Experiment/sample_data/Node/Cora/5_shot/1/train_idx.pt")

# GOOD
from prompt_graph.utils.paths import sample_dir
torch.save(obj, sample_dir('Node', 5, 'Cora') / '1' / 'train_idx.pt')
```

`prompt_graph/utils/paths.py` 暴露 `DATA_ROOT` / `EXPERIMENT_ROOT` 以及 `tudataset_root()` / `ogb_dataset_root()` / `induced_graph_dir()` / `sample_dir()` / `pretrained_model_dir()`。`PROG_DATA_ROOT` / `PROG_EXPERIMENT_ROOT` / `PROG_OGB_ROOT` 环境变量可以整体覆盖。

### 3.2 不要 `print()` 中间状态

只有"最终结果"（精度、Excel 写盘成功）保留 `print`。中间状态（epoch loss、early-stop、GNN summary、张量形状）一律走 logger：

```python
from prompt_graph.utils import get_logger
logger = get_logger(__name__)
logger.info("epoch %d loss=%.4f", epoch, loss)
logger.debug("z shape=%s", z.shape)
```

`--quiet` / `--log-level WARNING` 才能把这些静音。Phase 5.4 已经把全包 print 都转过来了，新引入 `print` 会回退覆盖。

### 3.3 不要直接 `torch.device(...)`

设备由 `prompt_graph.utils.resolve_device(device)` 统一解析。它处理：

- `int` → `cuda:N`（CUDA 不可用回退 MPS / CPU）；
- `'auto'` → 按 CUDA → MPS → CPU 优先级；
- `'cpu'` / `'mps'` / `'cuda:N'` → 显式；
- `torch.device(...)` → 原样返回。

不要再去读 `os.environ['PROG_USE_MPS']`，那是 Phase 3 之前的旧约定，已经删掉。

### 3.4 不要"破坏式"重命名公共 API

`ProG.tasker` / `ProG.model` 等公共 import 路径要保留 alias。重命名走两步：

1. 加新名 + 给老名挂 `DeprecationWarning`；
2. 下个 release 再删老名。

Phase 4 之前任何 PR 都不能改 `bench.py` 命令的输出 metric（误差 ≤ 1e-4 视为不变）。

### 3.5 `tests/` 里禁止 mock 数据集

集成 / smoke test 必须跑真 dataset（Cora / MUTAG 是最便宜的）。Phase 5.2 引入的 smoke test 暴露过 `range(1, self.epochs)` off-by-one、`int(epochs/answer_epoch)` 0 epoch、`load4node` 拼路径等真 bug —— mock 都掩盖不了。

### 3.6 OGB `torch.load` 别绕过

torch ≥ 2.4 默认 `weights_only=True`，会拒绝加载 OGB 的 `.pt`。统一走 `prompt_graph.data.load4data.ogb_torch_load_compat()` 这个 context manager：

```python
with ogb_torch_load_compat():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=...)
```

它做了 RLock 防止并发竞态把 `torch.load` 永远 monkey-patch 掉。新加 OGB 数据集时**必须**套这个 with。

---

## 4. 框架约束 / 历史决策

### 4.1 PromptStrategy 协议（Phase 4）

`prompt_graph/tasker/strategy.py` 定义 `PromptStrategy` Protocol、`TaskContext` dataclass、`STRATEGY_REGISTRY` 字典和 `register_strategy` 装饰器。每个 prompt_type 一个 strategy 类，通过 import 副作用注册（`strategies/__init__.py` 列出全部）。

加新 prompt_type 时严格按这个套路走，不要把分发逻辑塞回 `NodeTask.run` / `GraphTask.run` 的 if/elif。

### 4.2 `initialize_*` 当前还是 if/elif

截至 2026-05-12，`tasker/task.py:initialize_prompt` 和 `initialize_optimizer` 还在按 prompt_type 大 if/elif 分发；Phase 4 把 `run()` 已经切到 strategy，但这两个方法是 follow-up。改这两个方法时**两边都要改**（保留 if/elif，让现有 PR 不冲突），不要一刀切换掉。

### 4.3 NodeTask / GraphTask 返回值不对称

`AllInOneStrategy` 的 train_epoch：

- NodeTask 路径返回 `answer_loss`；
- GraphTask 路径返回 `pg_loss`。

doc §1.x 标注为"可疑但暂不动语义"，因为改动会让现有 baseline metric 漂移。如果你打算修，必须先在 [`Docs/baseline_metrics.md`](./Docs/baseline_metrics.md) 加一栏对照新基线。

### 4.4 `GraphTask.run` Branch A / B 双分支

`run()` 内部曾经有 few-shot (Branch A) 和 full-dataset (Branch B) 两路代码，几乎 1:1 重复但 All-in-one 的 epoch budget 不同（A=50/50、B=5/1）。Phase 4 Unit 21 已经合一，但保留两份返回值（9-tuple vs 4-tuple）。改这里时不要"为了优雅"把返回 tuple 长度统一掉，会破坏现有调用方。

### 4.5 `ENZYMES`、`PROTEINS` 节点分类的隐式约定

`load4node('ENZYMES')` 把多图 dataset 合成一个大图，并把节点特征**最后 3 列**当 one-hot label 用。`input_dim = dataset.num_node_features` 仍然包含那 3 列——这是历史代码的副作用。改这里之前先看 [`Docs/datasets.md`](./Docs/datasets.md) §5.1。

### 4.6 MPS 上的算子限制

Apple Silicon 的 MPS 后端不支持 `scatter_add_` 的 placeholder 路径。`GraphTask + All-in-one + MUTAG` 在 MPS 上会抛 `NotImplementedError`。修方法是显式 `--device cpu`；不要去 monkey-patch torch。

### 4.7 `LinkTask` 还未接入 strategy 框架

`prompt_graph/tasker/link_task.py` 存在但不属于 PromptStrategy 体系。顶层 README 上不要再写"支持 LinkTask"，直到 [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md) §1.10 被真正修复。

### 4.8 `GraphMultiGprompt` 预训练未实现

`pre_train.py:get_pretrain_task_delegate` 对 `GraphMultiGprompt` 显式抛 `NotImplementedError`，注释提到"等数据加载收敛后再补"。要碰这块前，先看 `load4graph(pretrained=True)` 的现状。

---

## 5. Phase 0 baseline 与 metric 漂移

任何代码改动都不应让 `bash scripts/baseline.sh` 的输出 metric 漂移超过 1e-4：

```bash
# 跑前
bash scripts/baseline.sh --tag before

# 改代码

# 跑后
bash scripts/baseline.sh --tag after

# diff Docs/baseline_metrics.md 中 before vs after 两栏
```

如果你 confident 改动**应当**影响 metric，把新栏写进 `Docs/baseline_metrics.md` 并在 PR description 解释为什么。

Strategy 重构 PR（Phase 4）的额外要求：跑 5 epoch 训练，**epoch 1-5 的 loss 序列与对照基线误差 ≤ 1e-3**，diff 表贴在 PR body。

---

## 6. 提交与分支

完整规范见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。简版：

- **分支命名**：`fix/<bug-id>` / `refactor/<phase>-<topic>` / `chore/<topic>` / `docs/<topic>` / `ci/<topic>`；
- **提交前缀**：`fix:` / `refactor:` / `chore:` / `docs:` / `test:` / `ci:`；
- **PR 基线**：默认 `--base dev`，不直接打 main；
- **一个 PR 一个目标**，方便回滚；Phase 1 是每条 bug 一个 PR，Phase 4 是每个 strategy 一个 PR。

---

## 7. 当你被卡住时

1. **现象 vs 测试**：先跑 `pytest tests/ -v`；smoke test 不挂大概率是你的本地环境出 issue。
2. **数据集下不下来**：检查 `PROG_OGB_ROOT` 是不是指错了；如果以前用旧默认 `./dataset` 现在升级了，需要 `export PROG_OGB_ROOT=$(pwd)/dataset` 指回去。
3. **改完 strategy 跑出来 loss 不一样**：先 diff `prompt_graph/tasker/task.py` 的 `initialize_optimizer`，看你的 prompt_type 是不是漏了一个 lr / wd；这是 Phase 4 迁移时最常见的回归原因。
4. **MPS 报错**：尝试 `--device cpu`；不行的话查 [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md) §1 看是不是已知的 MPS bug。
5. **CI 挂在 ruff**：本地 `ruff check . && ruff format .`，再 push。Ruff 配置在 `pyproject.toml`。

---

## 8. 不要轻易做的事

- **不要重写 `bench.py` 的 `param_grid`** —— baseline metric 漂移会牵涉所有历史对比；
- **不要删除老 prompt_type 的 strategy**，即使你认为它已经过时 —— 外部用户可能还在用；改走 deprecation 通道；
- **不要在 `.gitignore` 重新加 `/Docs`** —— Phase 6.1 已经决定把 `Docs/` 入库；
- **不要把 `Logo.jpg` / `ProG_pipeline.jpg` / `Node.zip` 重新提交到主仓库** —— Phase 5.6 已经从工作树移除，新版本走 Release Assets；
- **不要 mock `torch.load`、`torch.device` 或 `os.environ`** —— 测试要么真跑要么 `monkeypatch.setenv` / `monkeypatch.chdir`，全局 monkey-patch 会让其他 test 看到脏状态。

---

## 9. 进一步阅读

- [`Docs/architecture.md`](./Docs/architecture.md) — 模块边界
- [`Docs/datasets.md`](./Docs/datasets.md) — 数据集表 & 路径
- [`Docs/running.md`](./Docs/running.md) — 三个入口 + CLI
- [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md) — 改进 roadmap
- [`Docs/baseline_metrics.md`](./Docs/baseline_metrics.md) — Baseline metric 表
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — 协作者完整规范
