# GitHub Copilot Instructions for ProG

ProG is a graph prompt learning experiment framework. Core package: `prompt_graph/`. Three entry points: `pre_train.py`, `downstream_task.py`, `bench.py`.

For deeper detail, the authoritative project-level notes for AI agents live in [`CLAUDE.md`](../CLAUDE.md); human contributor rules are in [`CONTRIBUTING.md`](../CONTRIBUTING.md); architecture in [`Docs/architecture.md`](../Docs/architecture.md).

---

## 1. Build, lint, and test

```bash
# Dev install (Python 3.9 or 3.11; CI tests both)
pip install -e ".[dev]"
pre-commit install

# Lint (CI runs both; both must pass)
ruff check .
ruff format --check .       # use `ruff format .` to fix
# Ruff config: select = E,W,F,I,UP ; ignore = E501 ; line-length = 100

# Tests
pytest tests/ -v                                  # full suite (≈12 min on CPU)
pytest tests/test_strategy_gpf.py -v              # single file
pytest tests/test_strategy_gpf.py::test_name -v   # single test
pytest tests/ -k smoke -v                         # by keyword
pytest tests/ -m slow                             # opt-in RL-heavy cases
pytest tests/test_strategy_new_prompts.py -v      # 1-epoch coverage for all 17 strategies

# Baseline reproduction (don't let metrics drift > 1e-4)
bash scripts/baseline.sh                # full Phase-0 (3 frozen cases)
bash scripts/baseline.sh --fast         # ~50 epochs
bash scripts/baseline.sh --tag mytag    # logs to scripts/baseline_logs/

# Full-prompt coverage sweep (does NOT update baseline_metrics.md)
bash scripts/benchmark_all_prompts.sh --fast              # ~10-15 min
bash scripts/benchmark_all_prompts.sh --include-broken    # also run XFAIL combos
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, an import smoke on Python 3.9 + 3.11, and `pytest tests/ -v --tb=short -x` on 3.11. Local 3.11-pass ≠ CI pass — Python 3.9 rejects `X | Y` and `list[str]` typing syntax in runtime positions.

---

## 2. High-level architecture

Five core abstractions and where they live:

| Abstraction | Module | Entry |
|---|---|---|
| Dataset | `prompt_graph/data/` | `load4node` / `load4graph` → `(data, input_dim, output_dim)` |
| GNN backbone | `prompt_graph/model/` | `build_gnn` via registry in `__init__.py` |
| Pretrain paradigm | `prompt_graph/pretrain/` | `pre_train.py` dispatches 6 paradigms |
| Prompt module | `prompt_graph/prompt/` | `GPF` / `GPF_plus` / `Gprompt` / `HeavyPrompt` (All-in-one) / `GPPTPrompt` / `MultiGprompt` |
| Downstream task | `prompt_graph/tasker/` | `NodeTask` / `GraphTask`, dispatching via `PromptStrategy` |

### PromptStrategy protocol (Phase 4 refactor)

`prompt_graph/tasker/strategy.py` defines `PromptStrategy` Protocol, `TaskContext` dataclass, `STRATEGY_REGISTRY`, and `@register_strategy(name)`. Each prompt_type owns a class in `prompt_graph/tasker/strategies/`. Registration happens via import side effects in `strategies/__init__.py` (`from . import none, gpf, ...`).

Currently 16 strategies are registered (as of 2026-05-22): `None`, `GPF`, `GPF-plus`, `Gprompt`, `All-in-one`, `GPPT`, `MultiGprompt`, `Prodigy`, `UniPrompt`, `SelfPro`, `ProNoG`, `DAGPrompT`, `PSP`, `RELIEF`, `GraphPrompter`, `EdgePrompt` (+ `EdgePromptplus`). Treat `STRATEGY_REGISTRY.keys()` as ground truth — if this list has fallen behind, look at the registry directly. See [`Docs/architecture.md`](../Docs/architecture.md) §3 for the full table with paper references.

Adding a new prompt_type:

1. Add `strategies/<name>.py` with `@register_strategy('MyPrompt')` decorating a class implementing `setup` / `configure_optimizer` / `train_epoch` / `evaluate`.
2. Append `from . import <name>  # noqa: F401` to `strategies/__init__.py`.
3. Add a smoke test under `tests/test_strategy_<name>.py` (2-epoch run on Cora for NodeTask, MUTAG for GraphTask).

**Caveat (as of 2026-05):** `NodeTask.run` / `GraphTask.run` are on the strategy framework, but `tasker/task.py:initialize_prompt` and `initialize_optimizer` are still big `if/elif` chains by `prompt_type`. Touch both places when adding a strategy — don't unify them in passing.

### Init order in `BaseTask`

`__init__` → `resolve_device(device)` → `initialize_lossfn` → child `__init__` (saves data / dataset; MultiGprompt uses `load_multigprompt_data`) → `create_few_data_folder` → in `run()`: `load_pre_trained()` → `initialize_gnn()` → `initialize_prompt()` → `initialize_optimizer()` → epoch loop dispatches to `strategy.train_epoch` / `strategy.evaluate`.

---

## 3. Repository-specific conventions

### Paths — never hardcode `./data` / `./Experiment`

Always go through `prompt_graph.utils.paths`:

```python
from prompt_graph.utils.paths import sample_dir, induced_graph_dir, pretrained_model_dir
torch.save(obj, sample_dir('Node', 5, 'Cora') / '1' / 'train_idx.pt')
```

Roots are overridable: `PROG_DATA_ROOT`, `PROG_EXPERIMENT_ROOT`, `PROG_OGB_ROOT`. The old default for OGB was `./dataset`; upgraders may need `export PROG_OGB_ROOT=$(pwd)/dataset`.

### Logging — no `print` for intermediate state

```python
from prompt_graph.utils import get_logger
logger = get_logger(__name__)
logger.info("epoch %d loss=%.4f", epoch, loss)
```

`print()` is reserved for user-facing final results (`bench.py` "Final Accuracy", `create_excel_for_bench.py` "Data saved"). Everything else (epoch loss, early-stop, GNN summary, tensor shapes) goes through `logger`. CLI controls level with `--log-level {DEBUG,INFO,WARNING,ERROR}` or `--quiet` (= WARNING). Never call `logging.basicConfig` at module top-level.

### Device — never `torch.device(...)` directly

```python
from prompt_graph.utils import resolve_device
device = resolve_device(device_arg)   # int -> cuda:N (fallback MPS/CPU); 'auto'; 'cpu'; 'mps'; 'cuda:N'
```

Don't read `os.environ['PROG_USE_MPS']` — it's a removed pre-Phase-3 convention.

**MPS gotcha:** Apple Silicon's `scatter_add_` placeholder path is unsupported; `GraphTask + All-in-one + MUTAG` on MPS raises `NotImplementedError`. Use `--device cpu` for that combo.

### OGB `torch.load` compatibility

Torch ≥ 2.4 defaults to `weights_only=True` and refuses OGB's `.pt`. Always wrap OGB loads:

```python
from prompt_graph.data.load4data import ogb_torch_load_compat
with ogb_torch_load_compat():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=...)
```

This is an RLock-guarded context manager — don't reach past it with a manual `torch.load` monkey-patch.

### Tests — no dataset mocks, no global monkey-patches

Smoke / integration tests run real datasets (Cora / MUTAG are cheapest). Phase 5.2 smoke tests caught real bugs (`range(1, self.epochs)` off-by-one, `int(epochs/answer_epoch)` zero-epoch divide) that mocks would have hidden. Don't globally monkey-patch `torch.load` / `torch.device` / `os.environ`; use pytest's `monkeypatch.setenv` / `monkeypatch.chdir`.

### CLI args — three-layer precedence

`prompt_graph/utils/get_args.py:get_args` merges: defaults (`DEFAULT_ARG_DICT = vars(get_args_by_call())`) ← YAML (`--config <path>`) ← explicit CLI flags. Every non-meta `add_argument` uses `argparse.SUPPRESS`, so "user didn't pass" and "user passed default" are distinguishable. When adding a new CLI flag, update `_build_parser`, `get_args_by_call`, and `DEFAULT_ARG_DICT` together.

### Branch / commit / PR conventions

- Branch prefixes: `fix/<bug-id>` / `refactor/<phase>-<topic>` / `chore/<topic>` / `docs/<topic>` / `test/<topic>` / `ci/<topic>`.
- Commit prefixes match: `fix:` / `refactor:` / `chore:` / `docs:` / `test:` / `ci:`. Commit body explains **why**, not **what**.
- PRs default `--base dev` — never push directly to `main` (only `dev` → `main` merges hit main).
- One PR per goal. Phase 1 = one bug per PR; Phase 4 = one strategy per PR.

### Baseline metric drift

Any change should keep `bash scripts/baseline.sh` outputs within **1e-4** of `Docs/baseline_metrics.md`. Strategy refactor PRs additionally must match epoch 1-5 loss within **1e-3** of the baseline; paste the diff into the PR body. If you intentionally shift metrics, add a new column to `Docs/baseline_metrics.md` and justify it.

### Public API deprecation

`ProG.tasker` / `ProG.model` import aliases must be preserved. Renaming = two-step: add new name with `DeprecationWarning` on the old → delete in next release. Don't change `bench.py:param_grid` or its printed metrics without a baseline_metrics.md migration.

---

## 4. Project-specific quirks

- `load4node('ENZYMES')` collapses multi-graph datasets into one big graph and uses the **last 3 node-feature columns as one-hot labels**; `input_dim = dataset.num_node_features` still includes those 3 columns. Historical side effect — see `Docs/datasets.md` §5.1.
- `LinkTask` (`prompt_graph/tasker/link_task.py`) — file kept, but removed from the public API in commit `e76e20f` (no longer in `prompt_graph/tasker/__init__.__all__`). Not on the `PromptStrategy` framework. Don't re-export without finishing the §1.10 fix list in `Docs/IMPROVEMENTS.md`.
- `GraphMultiGprompt` pretraining is now wired up (commit `647d6c4`) — `pre_train.py:get_pretrain_task_delegate` calls `load4graph(args.dataset_name, pretrained=True)` then `GraphPrePrompt(...)`. Note `load4graph(pretrained=True)` returns `(input_dim, out_dim, graph_list)`, **different order** from the NodeMultiGprompt path.
- `AllInOneStrategy.train_epoch` returns `answer_loss` on the NodeTask path but `pg_loss` on the GraphTask path — asymmetric but intentionally not changed to preserve baseline metrics.
- `GraphTask.run` has two branches (few-shot Branch A, full-dataset Branch B) with different return tuple lengths (9-tuple vs 4-tuple). Don't "tidy" by unifying — callers depend on the shapes.
- Don't reintroduce `Logo.jpg` / `Node.zip` to the working tree (Phase 5.6 moved them to Release Assets; `ProG_pipeline.jpg` was later restored for README, commit `086bd8b`).
- Don't re-add `/Docs` to `.gitignore`; Phase 6.1 decided to commit it.

---

## 5. Further reading

- [`CLAUDE.md`](../CLAUDE.md) — full AI-agent conventions (anti-patterns, history, gotchas)
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — branching, PRs, review checklist
- [`Docs/architecture.md`](../Docs/architecture.md) — module boundaries + PromptStrategy
- [`Docs/datasets.md`](../Docs/datasets.md) — supported datasets + path quirks
- [`Docs/running.md`](../Docs/running.md) — three entry points + CLI flags + YAML configs
- [`Docs/IMPROVEMENTS.md`](../Docs/IMPROVEMENTS.md) — refactor roadmap, known bugs
- [`Docs/baseline_metrics.md`](../Docs/baseline_metrics.md) — metric snapshots
