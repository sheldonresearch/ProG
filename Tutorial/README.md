# ProG-V2 Quickstart Tutorial

This tutorial runs two minimal benchmark cells:

1. Cora node classification with no prompt/pretrain.
2. MUTAG graph classification with no prompt/pretrain.

Both use 1-shot splits, `GCN`, one random-search trial, and a configurable small
epoch budget. The goal is to verify that the public entry points, dataset
loaders, Excel result writers, and downstream tasks work end to end.

## Run

From the repository root:

```bash
python Tutorial/quickstart_progv2.py --device cpu --epochs 1
```

Use a GPU if available:

```bash
python Tutorial/quickstart_progv2.py --device cuda:0 --epochs 1
```

## Outputs

The tutorial writes bench-style matrices under:

```text
Experiment/ExcelResults/
```

These local experiment artifacts are ignored by git.
