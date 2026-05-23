# ProG 架构

本文档描述 ProG 仓库的模块边界、`PromptStrategy` 协议、初始化顺序，以及 `TaskContext` 的形状。目标读者是首次浏览代码、需要在某个 prompt_type / GNN / dataset 之上做扩展的工程师。

如果你只想"先把 baseline 跑起来"，请直接看 [`Docs/running.md`](./running.md)。

---

## 1. 模块层次

```
ProG/
├── prompt_graph/                # 核心包（PEP 621，可 pip install -e .）
│   ├── data/                    # 数据集加载与切分
│   ├── model/                   # GNN backbone 与 registry
│   ├── pretrain/                # 6 种预训练范式
│   ├── prompt/                  # 6 种 prompt 模块
│   ├── tasker/                  # 下游任务编排（含 PromptStrategy 框架）
│   ├── evaluation/              # 各 prompt_type 的评估实现
│   └── utils/                   # 日志 / 设备 / 损失 / 路径 / CLI args
├── configs/                     # YAML 配置（Phase 5.5）
├── tests/                       # pytest（CI 跑）
├── Docs/                        # 本目录
├── Tutorial/                    # 示例脚本（runnable）
├── scripts/baseline.sh          # 基线复现入口
├── pre_train.py                 # 预训练入口
├── downstream_task.py           # 单条下游任务入口
└── bench.py                     # 批量 / 网格搜索入口
```

`prompt_graph` 是唯一的"运行时"代码包。`Tutorial/` 是面向用户的演示；`scripts/`、`tests/`、`Docs/`、`configs/` 仅供工程协作。

---

## 2. 五个核心抽象

| 抽象 | 模块 | 入口 | 职责 |
|------|------|------|------|
| 数据集 | `prompt_graph.data` | `load4node` / `load4graph` | 返回 `(data, input_dim, output_dim)`；少样本切分由 `node_sample_and_save` / `graph_sample_and_save` 完成 |
| GNN backbone | `prompt_graph.model` | `build_gnn` | 通过 registry 注册（`GCN`, `GAT`, `GIN`, `GraphSAGE`, `GCov`, `GraphTransformer`） |
| 预训练范式 | `prompt_graph.pretrain` | `pre_train.py` | 6 种：`DGI` / `GraphCL` / `SimGRACE` / `Edgepred_GPPT` / `Edgepred_Gprompt` / `GraphMAE`（外加 `MultiGPrompt` 自带的 `NodePrePrompt` / `GraphPrePrompt`） |
| Prompt | `prompt_graph.prompt` | `GPF` / `GPF_plus` / `Gprompt` / `HeavyPrompt` (All-in-one) / `GPPTPrompt` / `MultiGprompt` | 在下游任务前缀拼到节点特征或 token 上 |
| 下游任务 | `prompt_graph.tasker` | `NodeTask` / `GraphTask` | 把 GNN + Prompt + Answering Head 串起来跑 train/eval；通过 `PromptStrategy` 分发 prompt_type 专属逻辑 |

不要把"评估实现"误认为"任务"：`evaluation/*.py` 只是给 tasker 调用的纯函数。

---

## 3. PromptStrategy 协议（Phase 4 重构产物）

Phase 4 之前，`NodeTask.run` / `GraphTask.run` 内部是按 `prompt_type` 分发的巨型 if/elif。这种结构改一个 prompt_type 就会 diff 出几十行不相关上下文，且很难独立测试。

Phase 4 把这一层拆成"策略"：

```python
# prompt_graph/tasker/strategy.py（节选）

@dataclass
class TaskContext:
    """传给 strategy 的可变状态包。"""
    gnn: Any = None
    prompt: Any = None
    answering: Any = None
    device: Any = None
    hid_dim: int = None
    output_dim: int = None
    criterion: Any = None
    optimizer: Any = None
    pg_opi: Any = None
    answer_opi: Any = None
    data: Any = None
    dataset_name: str = None
    extra: dict = field(default_factory=dict)


class PromptStrategy(Protocol):
    name: ClassVar[str]

    def setup(self, ctx: TaskContext) -> None: ...
    def configure_optimizer(self, ctx: TaskContext) -> torch.optim.Optimizer: ...
    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float: ...
    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple[float, float, float, float]: ...


STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(name: str):
    """装饰器：把 strategy 注册到 STRATEGY_REGISTRY[name]，重复注册抛 ValueError。"""
    ...


def get_strategy(name: str):
    """查询 strategy；未注册抛 KeyError，错误信息会列出所有已注册的 strategy。"""
    ...
```

### 当前已注册的 strategy（16 个，按注册时间）

| `prompt_type` | 类 | 文件 | 备注 |
|---|---|---|---|
| `None` | `NoneStrategy` | `strategies/none.py` | 无 prompt，纯标准微调 |
| `GPF` | `GPFStrategy` | `strategies/gpf.py` | GPF (Fang et al., 2022) |
| `GPF-plus` | `GPFPlusStrategy` | `strategies/gpf.py` | GPF+ (NeurIPS 2023) |
| `Gprompt` | `GpromptStrategy` | `strategies/gprompt.py` | GraphPrompt (WWW 2023) |
| `All-in-one` | `AllInOneStrategy` | `strategies/all_in_one.py` | KDD 2023 Best Paper |
| `GPPT` | `GPPTStrategy` | `strategies/gppt.py` | KDD 2022 |
| `MultiGprompt` | `MultiGpromptStrategy` | `strategies/multi_gprompt.py` | 与 `prompt/MultiGprompt.py` 强耦合 |
| `Prodigy` | `ProdigyStrategy` | `strategies/prodigy.py` | ICML 2023 SPIGM；commit `393cb08`，light-weight port |
| `UniPrompt` | `UniPromptStrategy` | `strategies/uni_prompt.py` | NeurIPS 2025 |
| `SelfPro` | `SelfProStrategy` | `strategies/self_pro.py` | ICML 2024 |
| `ProNoG` | `ProNoGStrategy` | `strategies/pro_no_g.py` | KDD 2025 |
| `DAGPrompT` | `DAGPrompTStrategy` | `strategies/dagprompt.py` | NeurIPS 2024 |
| `PSP` | `PSPStrategy` | `strategies/psp.py` | — |
| `RELIEF` | `RELIEFStrategy` | `strategies/relief.py` | NeurIPS 2024（MVP→full version commit `30401ca`） |
| `GraphPrompter` | `GraphPrompterStrategy` | `strategies/graph_prompter.py` | KDD 2025（MVP→full version commit `a2abd91`） |
| `EdgePrompt` / `EdgePromptplus` | `EdgePromptStrategy` | `strategies/edge_prompt.py` | 一个文件注册两个名字 |

> 注册通过 import 副作用完成 —— `prompt_graph/tasker/strategies/__init__.py` 仅做 `from . import none, gpf, ...`，"导入 tasker 包"等价于"注册全部 strategy"。`STRATEGY_REGISTRY` 的真实大小以 import 后的 `len()` 为准；本表如果又落后于代码，请直接查 `STRATEGY_REGISTRY.keys()`。

### 添加一个新 strategy

1. 在 `prompt_graph/tasker/strategies/` 下新建 `myprompt.py`：

   ```python
   from ..strategy import PromptStrategy, TaskContext, register_strategy

   @register_strategy('MyPrompt')
   class MyPromptStrategy:
       def setup(self, ctx: TaskContext) -> None: ...
       def configure_optimizer(self, ctx: TaskContext) -> torch.optim.Optimizer: ...
       def train_epoch(self, ctx: TaskContext, loader_or_data) -> float: ...
       def evaluate(self, ctx, loader_or_data): ...
   ```

2. 在 `strategies/__init__.py` 末尾追加 `from . import myprompt  # noqa: F401`。
3. 在 `tasker/task.py:initialize_prompt` 加入新分支构造 `self.prompt`（如果 prompt 实例是 strategy 之外构造的；不少 strategy 自己在 `setup` 里造）。
4. 写一份 `tests/test_strategy_myprompt.py`，至少覆盖：
   - 注册检查（`'MyPrompt' in STRATEGY_REGISTRY`）；
   - 一个 2 epoch 的 smoke run（NodeTask 用 Cora、GraphTask 用 MUTAG）。
5. **Phase 4 验证标准**：跑 5 epoch，与对照基线对比 epoch 1-5 的 loss 序列误差 ≤ 1e-3。

---

## 4. BaseTask 初始化顺序

`NodeTask` 和 `GraphTask` 都继承 `BaseTask`（`prompt_graph/tasker/task.py`）。`run()` 进入前，构造函数会按下列顺序设置状态：

```
1. __init__：保存 hyperparameters；解析 device（resolve_device）
2. initialize_lossfn：CrossEntropyLoss（Gprompt 改用 Gprompt_tuning_loss）
3. NodeTask/GraphTask 子类 __init__：保存 data / dataset；MultiGprompt 走 load_multigprompt_data
4. create_few_data_folder：写少样本切分到 Experiment/sample_data/...
5. run() 内：load_pre_trained() → initialize_gnn() → initialize_prompt() → initialize_optimizer()
6. run() 内循环：strategy.train_epoch / strategy.evaluate（或老的 elif 分支，正在迁移中）
```

> 注意：截至 2026-05-12，`NodeTask.run` / `GraphTask.run` 已切到 strategy，但 `initialize_prompt` / `initialize_optimizer` 还按 prompt_type 大 if/elif 分发。下一步是把这两个方法也搬进 strategy。这是 Phase 4 收尾的 follow-up，不阻塞当前 baseline。

### 设备解析

`BaseTask.__init__` 调 `prompt_graph.utils.resolve_device(device)`，接受：

- `int` —— `cuda:<int>` 如果 CUDA 可用，否则回退到 MPS / CPU；
- `'auto'` —— `cuda` > `mps` > `cpu`，按可用性挑第一个；
- `'cuda:N'` / `'mps'` / `'cpu'` —— 显式指定；
- `torch.device` —— 原样返回。

不要直接读 `os.environ['PROG_USE_MPS']`，那是 Phase 3 之前的旧约定，已删除。

---

## 5. 数据 / 实验 / OGB 路径

所有可写路径通过 `prompt_graph.utils.paths` 暴露：

| 常量 / 函数 | 默认 | 环境变量 |
|---|---|---|
| `DATA_ROOT` | `<repo>/data` | `PROG_DATA_ROOT` |
| `EXPERIMENT_ROOT` | `<repo>/Experiment` | `PROG_EXPERIMENT_ROOT` |
| `ogb_dataset_root()` | `<DATA_ROOT>/OGB` | `PROG_OGB_ROOT`（旧版默认 `./dataset`，升级时手动指回） |
| `induced_graph_dir(name)` | `<EXPERIMENT_ROOT>/induced_graph/<name>` | — |
| `sample_dir(task, k, name)` | `<EXPERIMENT_ROOT>/sample_data/<task>/<name>/<k>_shot` | — |
| `excel_result_dir(...)` | `<EXPERIMENT_ROOT>/ExcelResults/...` | — |
| `pretrained_model_dir(name)` | `<EXPERIMENT_ROOT>/pre_trained_model/<name>` | — |
| `tudataset_root()` | `<DATA_ROOT>/TUDataset` | — |
| `planetoid_root()` | `<DATA_ROOT>/Planetoid` | — |

代码里**不要拼字符串路径**；调上面的工厂函数。

---

## 6. 日志

```python
from prompt_graph.utils import get_logger
logger = get_logger(__name__)
logger.info("epoch %d loss=%.4f", epoch, loss)
```

- `print()` 只保留给 user-facing 终态（如 `bench.py` 的 "Final Accuracy"），其他统一走 logger；
- CLI 全局通过 `--log-level {DEBUG,INFO,WARNING,ERROR}` 或 `--quiet`（= WARNING）控制；
- 默认格式 `%(asctime)s %(levelname)s %(name)s - %(message)s`，由 `prompt_graph/utils/logging.py:get_logger` 统一注入；
- 不要在 module 顶层调 `logging.basicConfig`，会被 `bench.py` / `downstream_task.py` 的全局配置吃掉。

---

## 7. 测试与 CI

- 单元 / smoke：`tests/test_*.py`，pytest 跑；
- 6 个 strategy 各有一个 `tests/test_strategy_<name>.py` smoke test；
- CI（`.github/workflows/ci.yml`）：matrix `python-version: ['3.9', '3.11']`，跑 `ruff check` + `pytest`；
- 任何 strategy 改动都必须跑 [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) §6.0 的 5 步 e2e recipe，并贴 epoch 1-5 loss diff。

---

## 8. 相关文档

- [`Docs/datasets.md`](./datasets.md) — 数据集来源、根目录约定与特殊处理；
- [`Docs/running.md`](./running.md) — 三个入口（`pre_train` / `downstream_task` / `bench`）的命令行用法与配置策略；
- [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) — Phase 0-6 的重构 roadmap 与每个 Phase 的落地清单；
- [`Docs/baseline_metrics.md`](./baseline_metrics.md) — 基线 metric 快照（每次合并前 diff 用）；
- [`CLAUDE.md`](../CLAUDE.md) — 给 AI 协作者的项目级约定。
