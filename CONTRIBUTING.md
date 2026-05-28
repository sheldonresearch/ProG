# Contributing to ProG-V2

Thanks for contributing to ProG-V2. This guide describes the public development
workflow, coding conventions, and validation steps expected for pull requests.

## Development Setup

Use Python 3.9 or 3.11. Python 3.11 is recommended for local development.

```bash
conda create -n prog-v2 python=3.11 -y
conda activate prog-v2
pip install -e ".[dev]"
pre-commit install
```

Quick environment check:

```bash
python -c "import prompt_graph; print(prompt_graph.__file__)"
pytest tests/test_factory.py -v
```

## Branches and Commits

Use one branch per goal.

| Prefix | Use case | Example |
|---|---|---|
| `fix/` | Bug fixes | `fix/relief-node-eval-device` |
| `feat/` | New user-facing functionality | `feat/new-prompt-strategy` |
| `refactor/` | Internal structure changes | `refactor/strategy-optimizer-setup` |
| `test/` | Test additions or fixes | `test/backbone-registry` |
| `docs/` | Documentation updates | `docs/public-results` |
| `chore/` | Maintenance | `chore/update-ci` |

Commit messages should use the matching prefix:

```text
fix: handle WebKB multi-split train masks
docs: publish GCN overall performance report
test: cover strategy registration
```

Explain why a change is needed in the commit body when the reason is not obvious.

## Coding Conventions

### Paths

Do not hardcode `./data` or `./Experiment` in runtime code. Use helpers from
`prompt_graph.utils.paths`, such as:

- `excel_result_dir`
- `sample_dir`
- `pretrained_model_dir`
- `induced_graph_dir`

### Device handling

Use `prompt_graph.utils.resolve_device(device)` instead of constructing
`torch.device(...)` directly in task setup code.

### Logging

Use the project logger for intermediate state:

```python
from prompt_graph.utils import get_logger

logger = get_logger(__name__)
logger.info("epoch %d loss=%.4f", epoch, loss)
```

Keep `print()` for user-facing final results only.

### Type and compatibility notes

The project supports Python 3.9 and 3.11. Avoid runtime-only syntax that breaks
Python 3.9.

## Running Checks

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
pytest tests/ -v
```

For faster iteration:

```bash
pytest tests/test_factory.py -v
pytest tests/test_strategy_registry.py -v
pytest tests/test_strategy_gpf.py -v
```

## Adding a Prompt Strategy

1. Add a strategy implementation under `prompt_graph/tasker/strategies/`.
2. Register it with `@register_strategy("PromptName")`.
3. Import the module in `prompt_graph/tasker/strategies/__init__.py` so the
   registry is populated on package import.
4. Add or update initialization logic if the prompt needs custom modules.
5. Add a smoke test under `tests/`.

At minimum, a new strategy should be able to run a small Cora or MUTAG smoke
configuration.

## Adding a Backbone

1. Add the model implementation or wrapper under `prompt_graph/model/`.
2. Register it in the model factory used by `build_gnn`.
3. Add a construction + forward smoke test in `tests/test_factory.py`.
4. Verify at least one pretrain/downstream path with the new backbone.

## Adding a Dataset

1. Add loader support in `prompt_graph/data/load4data.py`.
2. Update dataset lists in `prompt_graph/defines.py`.
3. Add or update a data-loader smoke test.
4. Ensure generated files use `prompt_graph.utils.paths` helpers.

## Benchmark Results

Public merged reports should live under `results/`. Keep raw local outputs,
temporary merge workspaces, and machine-specific logs out of the repository.

The current public Overall Performance report is:

```text
results/overall-performance-gcn/
```

It contains a flat `summary.csv`, a `final_matrices.xlsx` workbook, and
bench-style per-dataset Excel matrices.

## Pull Request Checklist

- [ ] The PR has one clear goal.
- [ ] New runtime code avoids hardcoded paths.
- [ ] New task/model code uses centralized device handling.
- [ ] New intermediate output uses logging instead of `print()`.
- [ ] Tests were added or updated for new behavior.
- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] Relevant pytest targets pass.
- [ ] Public docs/results do not include local machine or private execution metadata.
