# 运行指南

本文档介绍 ProG 的三个运行入口、共享的 CLI 标志，以及 YAML 配置 / 设备选择 / 日志控制。

| 入口 | 用途 | 典型耗时 |
|---|---|---|
| `pre_train.py` | 跑 6 种预训练范式（DGI / GraphCL / SimGRACE / Edgepred_GPPT / Edgepred_Gprompt / GraphMAE / MultiGprompt）；产出 `Experiment/pre_trained_model/<dataset>/<task>.<gnn>.<hid>hidden_dim.pth` | 几分钟到几小时（数据规模 + epochs） |
| `downstream_task.py` | 单条 prompt-tuning 下游任务（NodeTask / GraphTask） | 几分钟 |
| `bench.py` | 多 prompt_type × 多 shot 的网格搜索；写 Excel 到 `Experiment/ExcelResults/` | 数小时 |

如果你只是想"跑一次基线"，看 [§7](#7-跑一次基线) 和 [`scripts/baseline.sh`](../scripts/baseline.sh)。

---

## 1. 共享 CLI 标志（来自 `prompt_graph/utils/get_args.py`）

三个入口都共享同一份 `_build_parser`：

| 标志 | 类型 | 默认 / 来源 | 说明 |
|---|---|---|---|
| `--config <path>` | str | None | YAML 配置文件路径；CLI 显式标志仍会覆盖 YAML |
| `--pretrain_task` | str | —— | 预训练范式 (`DGI` / `GraphCL` / `SimGRACE` / `Edgepred_GPPT` / `Edgepred_Gprompt` / `GraphMAE` / `MultiGprompt`) 或下游 task type (`NodeTask` / `GraphTask`)。在 `bench.py` / `pre_train.py` 都用 |
| `--downstream_task` | str | —— | `NodeTask` / `GraphTask` |
| `--dataset_name` | str | `Cora` | 详见 [`Docs/datasets.md`](./datasets.md) |
| `--gnn_type` | str | `GCN` | `GCN` / `GAT` / `GIN` / `GraphSAGE` / `GCov` / `GraphTransformer` |
| `--prompt_type` | str | `None` | 通过 `STRATEGY_REGISTRY` 注册。当前已注册 16 个：`None` / `GPF` / `GPF-plus` / `Gprompt` / `All-in-one` / `GPPT` / `MultiGprompt` / `Prodigy` / `UniPrompt` / `SelfPro` / `ProNoG` / `DAGPrompT` / `PSP` / `RELIEF` / `GraphPrompter` / `EdgePrompt` / `EdgePromptplus`。最新清单见 [`Docs/architecture.md`](./architecture.md) §3 或 `STRATEGY_REGISTRY.keys()` |
| `--hid_dim` | int | 128 | GNN 隐层维度 |
| `--num_layer` | int | 2 | GNN 层数 |
| `--epochs` | int | 1000 | 下游 / 预训练 epoch 数 |
| `--batch_size` | int | 128 | DataLoader batch |
| `--shot_num` | int | 1 | 每类训练样本数（k-shot） |
| `--task_num` | int | 5 | 重复 k-shot 切分跑几次取均值 |
| `--lr` | float | 0.001 | learning rate |
| `--decay` | float | 0 | weight decay |
| `--dropout_ratio` | float | 0.5 | dropout |
| `--graph_pooling` | str | `mean` | `sum` / `mean` / `max` / `set2set` / `attention` |
| `--JK` | str | `last` | `last` / `sum` / `max` / `concat` |
| `--pre_train_model_path` | str | `None` | 预训练权重路径；`'None'`（字符串）= 从头训 |
| `--pnum` | int | 5 | GPF-plus 基数 |
| `--seed` | int | 42 | 数据切分种子 |
| `--runseed` | int | 0 | 训练循环种子 |
| `--num_workers` | int | 0 | DataLoader worker 数 |
| `--device` | str | 0 | int 是 CUDA index；也接 `auto` / `cuda:N` / `mps` / `cpu` |
| `--log-level` | str | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--quiet` | flag | False | 等价 `--log-level WARNING` |

> 所有非 `--config` / `--log-level` / `--quiet` 的标志都用 `argparse.SUPPRESS` 作为默认，所以"用户没传"和"用户传了默认值"在合并 YAML 时是不同的。详见 [§3](#3-yaml-配置cli--yaml--默认值)。

---

## 2. 设备 (`--device`)

`prompt_graph.utils.resolve_device(device)` 是统一入口：

| 传入 | 行为 |
|---|---|
| `0`, `1`, ... (`int`) | `cuda:N`，CUDA 不可用回退 MPS 再回退 CPU |
| `'auto'` | 按 CUDA → MPS → CPU 优先级挑可用的 |
| `'cuda:0'` / `'cuda:1'` | 显式 CUDA |
| `'mps'` | Apple Silicon GPU |
| `'cpu'` | 强制 CPU |
| `torch.device(...)` | 原样使用 |

**注意 MPS 限制**：少数算子（如 `scatter_add_` 的 placeholder 路径）在 MPS 上不支持，会抛 `NotImplementedError`。`GraphTask + All-in-one` 在 MUTAG 上目前需要 `--device cpu`。Bug 跟踪在 [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) §1。

---

## 3. YAML 配置（CLI > YAML > 默认值）

合并顺序（在 `prompt_graph/utils/get_args.py:get_args` 实现）：

1. **默认值**：`DEFAULT_ARG_DICT = vars(get_args_by_call())`，等价于不传任何参数时 `get_args_by_call` 的返回值；
2. **YAML 覆盖默认值**：`yaml.safe_load(open(args.config))` 的键值合并进去；
3. **CLI 显式标志覆盖 YAML**：因为所有 add_argument 用 `SUPPRESS`，只有用户在命令行写出来的标志才会出现在 `args_ns`。

示例：

```bash
# 跑 baseline 配置：
python bench.py --config configs/cora_gpf.yaml

# 同一份配置，但覆盖 epochs：
python bench.py --config configs/cora_gpf.yaml --epochs 50

# 不传 --config，全走 CLI：
python bench.py --pretrain_task NodeTask --dataset_name Cora \
                --prompt_type GPF --gnn_type GCN --shot_num 5 --seed 42 \
                --epochs 200 \
                --pre_train_model_path ./Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth
```

`configs/` 已有的 baseline 配置：

- [`configs/cora_gpf.yaml`](../configs/cora_gpf.yaml) — `Cora + GraphCL + GPF`（NodeTask）
- [`configs/mutag_allinone.yaml`](../configs/mutag_allinone.yaml) — `MUTAG + GraphCL + All-in-one`（GraphTask）
- [`configs/pubmed_gprompt.yaml`](../configs/pubmed_gprompt.yaml) — `PubMed + GraphCL + Gprompt`（NodeTask）

YAML 顶层必须是 mapping；不是的话 `_load_yaml_config` 会抛 `ValueError`。

---

## 4. `pre_train.py` — 预训练入口

```bash
# 用 GraphCL 在 Cora 上跑 200 epoch GCN
python pre_train.py --pretrain_task GraphCL --dataset_name Cora --gnn_type GCN \
                    --hid_dim 128 --num_layer 2 --epochs 200 --device 0
```

产出：`Experiment/pre_trained_model/<dataset>/<task>.<gnn>.<hid>hidden_dim.pth`。

支持的 `--pretrain_task`：

- `DGI` / `GraphCL` / `SimGRACE` / `Edgepred_GPPT` / `Edgepred_Gprompt` / `GraphMAE`
- `MultiGprompt` / `NodeMultiGprompt` / `GraphMultiGprompt`：自 commit `647d6c4` 起，`GraphMultiGprompt`（或当 `dataset_name in GRAPH_TASKS` 时的 `MultiGprompt`）走 `load4graph(args.dataset_name, pretrained=True)` → `GraphPrePrompt(...)` 真实路径，不再 `NotImplementedError`。

`get_pretrain_task_by_dataset_name` 会按 dataset 自动推断是 NodeTask 还是 GraphTask；图级数据集走 `GraphPrePrompt` / `Graph*` 实现。

---

## 5. `downstream_task.py` — 单条下游任务

```bash
# NodeTask
python downstream_task.py \
  --pretrain_task NodeTask --downstream_task NodeTask \
  --dataset_name Cora --gnn_type GCN \
  --prompt_type GPF \
  --shot_num 5 --epochs 200 --seed 42 --device 0 \
  --pre_train_model_path ./Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth

# GraphTask（MUTAG，注意 MPS 限制）
python downstream_task.py \
  --pretrain_task GraphTask --downstream_task GraphTask \
  --dataset_name MUTAG --gnn_type GCN \
  --prompt_type All-in-one \
  --shot_num 5 --epochs 200 --seed 42 --device cpu \
  --pre_train_model_path ./Experiment/pre_trained_model/MUTAG/GraphCL.GCN.128hidden_dim.pth
```

`downstream_task.py:get_downstream_task_delegate` 负责构造 `NodeTask` / `GraphTask`；induced-graph 缓存在 `Experiment/induced_graph/<dataset>/`。

---

## 6. `bench.py` — 网格搜索

```bash
python bench.py --config configs/cora_gpf.yaml
```

或者完全用 CLI：

```bash
python bench.py --pretrain_task NodeTask --dataset_name Cora --gnn_type GCN \
                --prompt_type GPF --shot_num 5 --seed 42 --epochs 200 \
                --pre_train_model_path ./Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth
```

`bench.py` 会枚举 `param_grid`（写在脚本里），对每个组合跑 `task_num` 个 k-shot 切分并把结果写进 `Experiment/ExcelResults/<task>/<shot>shot/<dataset>/...`。Excel 文件由 `create_excel_for_bench.py` 预生成（baseline.sh 会先调一次）。

---

## 7. 跑一次基线

```bash
# 完整 baseline（耗时较长）
bash scripts/baseline.sh

# 快速回归（缩短 epochs）
bash scripts/baseline.sh --fast

# 给本次 run 一个标签，写到 Docs/baseline_metrics.md
bash scripts/baseline.sh --tag phase-6
```

`scripts/baseline.sh` 内部就是上面三份 YAML 对应的命令，同时把 stdout 落盘到 `scripts/baseline_logs/<tag>_<datetime>.log`。每个 Phase 合并前都应跑一遍，把 metric 对照[`Docs/baseline_metrics.md`](./baseline_metrics.md) 更新。

> 注意：`baseline.sh` 是 **Phase-0 金标准**，case 集合是冻结的，metric 漂移 > 1e-4 会被 review push back。**不要往里加 case**——所有新增 prompt 方法的覆盖跑请用下面 §7.1 的 `benchmark_all_prompts.sh`。

### 7.1 全 prompt 覆盖跑（`scripts/benchmark_all_prompts.sh`）

```bash
# 跑所有"目前能正常工作"的 prompt × {NodeTask Cora, GraphTask MUTAG}（默认 200 epoch）
bash scripts/benchmark_all_prompts.sh

# 快速回归（50 epoch，~5-10 分钟）
bash scripts/benchmark_all_prompts.sh --fast

# 给本次 run 一个 tag
bash scripts/benchmark_all_prompts.sh --tag dev-2026-05

# 同时跑当前 XFAIL 的组合（参考 tests/test_strategy_new_prompts.py 已知 bug 列表），
# 用来验证你刚修的 prompt fix 真的把 SKIP/FAIL 转成 PASS
bash scripts/benchmark_all_prompts.sh --include-broken
```

脚本会把每个 case 的 PASS/FAIL/SKIP 汇总到 `scripts/baseline_logs/<tag>_<stamp>_all_prompts.log`。失败不会 abort sweep（区别于 `baseline.sh` 的 `set -e`），所以你能一次性看到全部覆盖结果。case 列表写在脚本里，不抽成 YAML —— 每加一个 strategy / 修一个 bug 就直接 diff 这个 shell 文件，可读性更高。

对应的"快速 smoke"是 `pytest tests/test_strategy_new_prompts.py`，1 epoch、~30 秒、用 `pytest.mark.xfail` 守护已知 bug；CI 跑的就是它，不跑全 sweep。

---

## 8. 日志控制

```bash
# 全开 DEBUG（包括张量形状、early-stop 决策等）
python downstream_task.py --config configs/cora_gpf.yaml --log-level DEBUG

# 静音（只剩 RESULT print 与 WARNING+ 日志）
python downstream_task.py --config configs/cora_gpf.yaml --quiet
```

`bench.py` / `downstream_task.py` 入口会调 `prompt_graph.utils.apply_log_level(args.log_level, args.quiet)`，统一把 root logger 的级别调到对应阈值。

> `print()` 只保留给"最终精度" / "结果写盘成功"等 user-facing 终态行，会**绕过** `--quiet`；epoch loss / early-stop / GNN summary 等中间状态都是 logger 调用，会被 `--quiet` 静音。

---

## 9. 相关文档

- [`Docs/architecture.md`](./architecture.md) — 三个入口背后的模块结构与 PromptStrategy 协议；
- [`Docs/datasets.md`](./datasets.md) — `--dataset_name` 支持哪些以及踩坑点；
- [`Docs/IMPROVEMENTS.md`](./IMPROVEMENTS.md) — 已知 bug、deprecated 标志、roadmap；
- [`Docs/baseline_metrics.md`](./baseline_metrics.md) — 历次 baseline 跑出的指标快照；
- [`scripts/baseline.sh`](../scripts/baseline.sh) — Phase 0 金标准命令。
- [`scripts/benchmark_all_prompts.sh`](../scripts/benchmark_all_prompts.sh) — 全 prompt 覆盖 sweep（§7.1）。
- [`tests/test_strategy_new_prompts.py`](../tests/test_strategy_new_prompts.py) — 全 prompt 的 1-epoch pytest smoke（带 XFAIL 已知 bug 列表）。
