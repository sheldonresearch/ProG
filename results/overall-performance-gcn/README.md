# Overall Performance Results

This directory contains the public merged Overall Performance report for ProG-V2.
The report covers a representative GCN benchmark grid over node- and graph-level
few-shot tasks.

## Scope

- Backbone: GCN
- GNN layers: 2
- Hidden dimension: 128
- Seed: 42
- Shots: 1-shot, 3-shot, 5-shot
- Metrics: Accuracy, Macro-F1, AUROC

The final table contains **714 independent `(dataset, shot, pretrain+prompt)`
combinations** and **2142 metric values**.

## Files

| File / directory | Description |
|---|---|
| `summary.csv` | Flat table. Each row is one `(task, shot, dataset, combo)` entry with `Final Accuracy`, `Final F1`, and `Final AUROC`. |
| `final_matrices.xlsx` | Workbook with 12 non-empty sheets: 4 datasets × 3 shots. |
| `{Node,Graph}/{1,3,5}shot/{dataset}/GCN_total_results.xlsx` | Matrix format compatible with `bench.py`: rows are metrics, columns are `pretrain+prompt` combinations. |

## Coverage

| Task | Dataset | 1-shot combos | 3-shot combos | 5-shot combos |
|---|---|---:|---:|---:|
| Node | Cora | 72 | 72 | 72 |
| Node | Wisconsin | 59 | 59 | 59 |
| Graph | MUTAG | 56 | 56 | 56 |
| Graph | PROTEINS | 51 | 51 | 51 |

Totals:

- Cora: 216 combinations
- Wisconsin: 177 combinations
- MUTAG: 168 combinations
- PROTEINS: 153 combinations
- Overall: **714 combinations**

## Pretraining Methods

The merged report includes:

- `None`
- `DGI`
- `GraphCL`
- `GraphMAE`
- `Edgepred_Gprompt`
- `MultiGprompt`

Pairing rules:

- `None` prompt is paired only with `None` pretrain.
- `MultiGprompt` prompt is paired only with `MultiGprompt` pretrain.
- Other prompt methods are paired with `None`, `DGI`, `GraphCL`, `GraphMAE`, and `Edgepred_Gprompt`.

## Prompt Strategies

The merged report includes 17 prompt strategies:

`None`, `GPF`, `GPF-plus`, `Gprompt`, `All-in-one`, `GPPT`, `Prodigy`,
`GraphPrompter`, `EdgePrompt`, `EdgePromptplus`, `RELIEF`, `MultiGprompt`,
`UniPrompt`, `SelfPro`, `ProNoG`, `PSP`, `DAGPrompT`.

Task applicability:

| Scope | Prompt strategies |
|---|---|
| Node + Graph | `None`, `GPF`, `GPF-plus`, `Gprompt`, `All-in-one`, `GPPT`, `Prodigy`, `GraphPrompter`, `EdgePrompt`, `EdgePromptplus`, `RELIEF` |
| Node-only | `MultiGprompt`, `UniPrompt`, `SelfPro`, `ProNoG`, `PSP` |
| Graph-only | `DAGPrompT` |

## RELIEF Coverage

The node-level RELIEF results are included for both Cora and Wisconsin. Each shot
contains the following five RELIEF columns:

- `None+RELIEF`
- `DGI+RELIEF`
- `GraphCL+RELIEF`
- `GraphMAE+RELIEF`
- `Edgepred_Gprompt+RELIEF`

## Metric Definitions

Each cell is reported as `mean±std` over 5 few-shot splits.

### Final Accuracy

Classification accuracy on the test split:

```text
Accuracy = correct predictions / test samples
```

### Final F1

Macro-F1. F1 is computed per class and then averaged across classes without
class-frequency weighting.

### Final AUROC

Area under the ROC curve. For multiclass tasks, the implementation uses the
multiclass/macro-style aggregation provided by the task strategy or metric
helper.

### AUPRC

The task code can compute AUPRC internally, but the public result matrices keep
the same three-row format used by `bench.py`: Accuracy, F1, and AUROC.

## Notes

- This report is a representative ProG-V2 Overall Performance sweep, not an
  exhaustive run over every dataset/backbone combination.
- The public report currently uses GCN. Other backbones are available through
  the model registry but are not included in this merged Overall Performance
  table.
- The result files intentionally contain only merged metrics and no raw training
  logs or machine-specific execution metadata.
