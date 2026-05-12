# Contributing to ProG

欢迎给 ProG 提 PR！本文是给**人类协作者**的完整协作规范。给 AI agent 的项目级约定在 [`CLAUDE.md`](./CLAUDE.md)；总体架构在 [`Docs/architecture.md`](./Docs/architecture.md)；改进 roadmap 在 [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md)。

---

## 1. 开发环境

推荐用 conda：

```bash
conda create -n ProG python=3.11 -y
conda activate ProG
pip install -e ".[dev]"
pre-commit install
```

`pip install -e ".[dev]"` 会装上 `pytest` / `ruff` / `pre-commit` / `mypy`。生产依赖在 `pyproject.toml:[project.dependencies]`，按需添加可选依赖（`[ogb]` 等）。

跑一次 smoke 确认环境 OK：

```bash
pytest tests/ -v
python -c "import prompt_graph; print(prompt_graph.__file__)"
```

更细的入口说明见 [`Docs/running.md`](./Docs/running.md)。

---

## 2. 分支与提交

### 2.1 分支命名

| 前缀 | 用途 | 示例 |
|---|---|---|
| `fix/<bug-id>` | 修 bug，bug-id 一般是 IMPROVEMENTS.md 的小节号 | `fix/1.2-multigprompt-nameerror` |
| `refactor/<phase>-<topic>` | 重构 / Phase 任务 | `refactor/phase4-allinone-strategy` |
| `chore/<topic>` | 维护性工作（依赖升级、配置） | `chore/bump-torch-2.4` |
| `docs/<topic>` | 文档 | `docs/improvements-table-phase4-5-done` |
| `test/<topic>` | 测试新增 / 修复 | `test/strategy-smoke` |
| `ci/<topic>` | CI / workflow | `ci/add-ruff-format-check` |

### 2.2 提交信息

提交头要前缀化，与分支前缀对应：`fix:` / `refactor:` / `chore:` / `docs:` / `test:` / `ci:`。

```
docs(phase6): split Docs/README.md into architecture/datasets/running

- 把"模块边界与初始化顺序"挪到 Docs/architecture.md
- 把"数据集表 + 路径约定"挪到 Docs/datasets.md
- 把"CLI 标志 + 三个入口"挪到 Docs/running.md
- Docs/README.md 收敛为导航 hub
```

正文 ≤ 72 字符宽度。**为什么**比**什么**重要——diff 已经告诉了 reviewer 改了啥，commit body 要解释为什么这么改。

### 2.3 一个 PR 一个目标

每个 PR 只解决一个独立目标，方便回滚。如果同时修两个 bug、又顺便重命名一个变量，请拆成三个 PR。Phase 1 是 "每条 bug 一个 PR"；Phase 4 是 "每个 strategy 一个 PR"。

例外：纯机械的批量改动（如 `print → logger`、`ruff --fix`）可以一个 PR 包一个目录。

---

## 3. Pull Request

### 3.1 基线分支

**默认 `--base dev`**，不要直接打到 `main`。`main` 只接受 `dev` → `main` 的发版合并。

```bash
gh pr create --base dev --title "fix(1.2): ..." --body "..."
```

### 3.2 PR 描述模板

```markdown
## 背景
（为什么要做这个改动；如果是修 bug，链接到 IMPROVEMENTS.md 的小节或 issue）

## 改动
（高层次描述，对应 commit messages 的总和；不必逐文件列）

## 验证
（每个 PR 必跑的最小验证集）
- [ ] `ruff check .` 通过
- [ ] `pytest tests/ -v` 通过
- [ ] （如有 NodeTask 影响）`python downstream_task.py --dataset_name Cora --gnn_type GCN --prompt_type GPF --epochs 2 --shot_num 5 --seed 42 --device 0`
- [ ] （如有 GraphTask 影响）`python downstream_task.py --dataset_name MUTAG --gnn_type GCN --prompt_type All-in-one --epochs 2 --shot_num 5 --seed 42 --device cpu`

## Baseline diff
（如有 metric 漂移，贴 Docs/baseline_metrics.md 的 before/after diff；如未漂移，写"无"）
```

### 3.3 review 关注点

reviewer 应当 push back 的情况：
- 没有"一个 PR 一个目标"——请求拆分；
- 引入了 [`CLAUDE.md`](./CLAUDE.md) §3 列出的反模式（硬编码路径 / `print` / 直接 `torch.device(...)`）；
- 改了公共 API 没走 deprecation 通道（见 [`CLAUDE.md`](./CLAUDE.md) §3.4）；
- baseline metric 漂移但没在 `Docs/baseline_metrics.md` 备案；
- 测试 mock 了 dataset 或全局 `torch.load`。

---

## 4. 代码风格

### 4.1 Ruff

`pyproject.toml` 配置：`select = ["E", "W", "F", "I", "UP"]`，`line-length=100`，`ignore = ["E501"]`。

本地：

```bash
ruff check .
ruff format .
```

`pre-commit` 在 commit 时自动跑一遍（`pre-commit install` 后生效）。

### 4.2 日志

不要 `print` 中间状态——一律 `from prompt_graph.utils import get_logger` 然后 `logger.info(...)` / `logger.debug(...)`。详见 [`CLAUDE.md`](./CLAUDE.md) §3.2 与 [`Docs/architecture.md`](./Docs/architecture.md) §6。

只有 "最终结果" 这种 user-facing 终态 print 才保留（`bench.py` 的 `Final Accuracy`、`create_excel_for_bench.py` 的 `Data saved`）。

### 4.3 路径

不要硬编码 `./data` / `./Experiment`——用 `prompt_graph.utils.paths` 暴露的工厂函数。详见 [`CLAUDE.md`](./CLAUDE.md) §3.1。

### 4.4 设备

不要直接 `torch.device(...)` 或读 `os.environ['PROG_USE_MPS']`（后者是 Phase 3 之前的老约定，已删除）。用 `prompt_graph.utils.resolve_device(device)` 统一解析。详见 [`Docs/running.md`](./Docs/running.md) §2。

---

## 5. 测试

### 5.1 跑测试

```bash
pytest tests/ -v             # 全量
pytest tests/test_strategy_gpf.py -v   # 单个文件
pytest tests/ -k smoke -v    # 只跑带 'smoke' 字样的
```

### 5.2 测试规约

- **不 mock dataset**：smoke / integration test 必须跑真 dataset（Cora / MUTAG 是最便宜的）。Phase 5.2 引入的 smoke test 暴露过 `range(1, self.epochs)` off-by-one 等真 bug，mock 都掩盖不了。详见 [`CLAUDE.md`](./CLAUDE.md) §3.5。
- **不全局 monkey-patch**：`torch.load` / `torch.device` / `os.environ` 都不要全局打 monkey patch；用 `monkeypatch.setenv` / `monkeypatch.chdir` 等 pytest fixtures。
- **每个新 prompt_type / strategy 都要带 smoke test**：至少一个 2-epoch NodeTask Cora 跑、GraphTask 必要时配一个。
- **改 strategy 必须跑 loss-diff 校验**：5 epoch 训练，epoch 1-5 loss 与 dev 基线误差 ≤ 1e-3。在 PR body 贴 diff 表。

### 5.3 CI

`.github/workflows/ci.yml`：matrix `python-version: ['3.9', '3.11']`，跑 `ruff check` + `pytest`。本地通过不代表 CI 通过——尤其是 3.9 上某些 typing 语法（`X | Y`、`list[str]`）会挂。

---

## 6. Baseline metric 漂移

任何代码改动都不应让 `bash scripts/baseline.sh` 的输出 metric 漂移超过 **1e-4**。

```bash
# 跑前
bash scripts/baseline.sh --tag before-myfix

# 改代码

# 跑后
bash scripts/baseline.sh --tag after-myfix

# diff Docs/baseline_metrics.md 中 before vs after 两栏
```

如果你 confident 改动**应当**影响 metric（例如修了 §1 列出的某个 bug），把新栏写进 `Docs/baseline_metrics.md` 并在 PR description 解释原因，同时通知正在用旧基线写论文的同事。

Strategy 重构 PR（Phase 4）的额外要求：跑 5 epoch 训练，**epoch 1-5 的 loss 序列与对照基线误差 ≤ 1e-3**。

`--fast` 标志会把 epochs 缩到 50，适合本地快速回归。完整 baseline（耗时较长）请在 review 前跑一次。

---

## 7. 添加新东西

| 添加什么 | 入口 | 关联 doc |
|---|---|---|
| 新 prompt_type | `prompt_graph/tasker/strategies/` 加一个 strategy 类 + `strategies/__init__.py` 注册 | [`Docs/architecture.md`](./Docs/architecture.md) §3 |
| 新 GNN backbone | `prompt_graph/model/__init__.py` registry + `build_gnn` | [`Docs/architecture.md`](./Docs/architecture.md) §2 |
| 新数据集 | `prompt_graph/data/load4data.py` + `prompt_graph/defines.py:NODE_TASKS/GRAPH_TASKS` | [`Docs/datasets.md`](./Docs/datasets.md) §1-2 |
| 新 CLI 标志 | `prompt_graph/utils/get_args.py:_build_parser` + `get_args_by_call` + `DEFAULT_ARG_DICT` | [`Docs/running.md`](./Docs/running.md) §1 |
| 新预训练范式 | `prompt_graph/pretrain/` + `pre_train.py:get_pretrain_task_delegate` | [`Docs/running.md`](./Docs/running.md) §4 |

---

## 8. Deprecation 与公共 API

`ProG.tasker` / `ProG.model` 等公共 import 路径要保留 alias，外部 user 可能在引用。重命名 / 删除走两步：

1. 加新名 + 给老名挂 `DeprecationWarning`；
2. 下个 release 再删老名（在 CHANGELOG 标记 breaking）。

`bench.py` 的 `param_grid` / 命令行输出 metric 不能擅自改——影响所有历史对比。如有需要，先在 [`Docs/baseline_metrics.md`](./Docs/baseline_metrics.md) 加新基线栏。

---

## 9. 报 bug / 提 feature

- 优先在 `Docs/IMPROVEMENTS.md` 里查是否已有 follow-up；
- 新 bug 在 GitHub Issues 开 issue，引用代码位置（`file_path:line_number`）；
- 想做 follow-up 改进的，欢迎在 PR description 里链 issue / IMPROVEMENTS.md 小节。

---

## 10. 相关文档

- [`CLAUDE.md`](./CLAUDE.md) — 给 AI 协作者的项目级约定（精简版）
- [`Docs/architecture.md`](./Docs/architecture.md) — 模块边界 + PromptStrategy 协议
- [`Docs/datasets.md`](./Docs/datasets.md) — 数据集表与路径约定
- [`Docs/running.md`](./Docs/running.md) — 三个入口的 CLI 与配置
- [`Docs/IMPROVEMENTS.md`](./Docs/IMPROVEMENTS.md) — Phase 0-6 重构 roadmap
- [`Docs/baseline_metrics.md`](./Docs/baseline_metrics.md) — Baseline metric 快照
