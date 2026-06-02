# Benchmark Results (Edge Task / Link Prediction)

This directory contains the public merged **edge-task** report for ProG-V2,
produced by the `LinkTask` downstream task. The companion node- and
graph-classification report lives under
[`../benchmark-gcn/`](../benchmark-gcn/).

Link prediction scores candidate node pairs, trains them with binary labels, and
is evaluated mainly by **AUROC** and **AUPRC**. Most prompt strategies use a
dot-product edge decoder. `All-in-one` follows its paper formulation: each
candidate edge `(u, v)` is reformulated into an edge-induced subgraph that is
classified by the edge label (see
`prompt_graph/tasker/all_in_one_link_adapter.py`).

## Scope

- Backbone: GCN
- GNN layers: 2
- Hidden dimension: 128
- Seed: 42
- Shots: 0-shot, 1-shot, 3-shot, 5-shot
- Metrics: Accuracy, F1, AUROC, AUPRC

`shot_num=0` is the full `RandomLinkSplit` baseline; `shot_num>0` is the
k-positive few-shot setting.

### Run configuration

- Downstream epochs: 50
- Folds per cell (`task_num`): 3 — every `mean±std` is taken over 3 independent
  few-shot splits, so the reported std reflects real run-to-run variance.
- Pretraining: 50 epochs per method.
- `All-in-one` edge-induced-subgraph caps: `max_train_edges=128`, `num_hops=1`,
  `max_nodes=24` (bounded so the edge-classification reformulation stays
  tractable on the graph-level datasets).

The final report contains **2912 independent `(dataset, shot, pretrain+prompt)`
combinations** and **11648 metric values** over Accuracy, F1, AUROC, and AUPRC.

## Files

| File / directory | Description |
|---|---|
| `summary.csv` | Flat table. Each row is one `(task, shot, dataset, combo)` entry with `Final Accuracy`, `Final F1`, `Final AUROC`, and `Final AUPRC`. |
| `final_matrices.xlsx` | Workbook with 32 non-empty sheets: 8 datasets × 4 shots. |
| `Link/{0,1,3,5}shot/{dataset}/GCN_total_results.xlsx` | Matrix format compatible with `bench.py`: rows are metrics, columns are `pretrain+prompt` combinations. |

## Coverage

| Dataset | 0-shot | 1-shot | 3-shot | 5-shot |
|---|---:|---:|---:|---:|
| CiteSeer | 91 | 91 | 91 | 91 |
| Cora | 91 | 91 | 91 | 91 |
| IMDB-BINARY | 91 | 91 | 91 | 91 |
| MUTAG | 91 | 91 | 91 | 91 |
| PROTEINS | 91 | 91 | 91 | 91 |
| PTC_MR | 91 | 91 | 91 | 91 |
| PubMed | 91 | 91 | 91 | 91 |
| Wisconsin | 91 | 91 | 91 | 91 |

Each `(dataset, shot)` cell holds **91 combinations** = 13 prompts × 7 pretrains,
for **364 combinations per dataset** and **2912 in total**.

## Pretraining Methods

The merged report includes 7 pretrains:

- `None` (train from scratch)
- `DGI`
- `GraphMAE`
- `Edgepred_GPPT`
- `Edgepred_Gprompt`
- `GraphCL`
- `SimGRACE`

Unlike the classification report, LinkTask keeps every `{pretrain}+None` cell,
since a bare GNN with no prompt is a meaningful link-prediction baseline.

## Prompt Strategies

The merged report includes the 13 LinkTask-supported prompt strategies:

`None`, `All-in-one`, `GPF`, `GPF-plus`, `EdgePrompt`, `EdgePromptplus`,
`UniPrompt`, `Gprompt`, `SelfPro`, `ProNoG`, `PSP`, `DAGPrompT`, `GraphPrompter`.

`GPPT`, `MultiGprompt`, `Prodigy`, and `RELIEF` are not yet adapted to LinkTask
and raise `NotImplementedError` at construction.

## Metric Definitions

Each cell is reported as `mean±std` over the configured few-shot splits.

### Final AUROC / Final AUPRC

Area under the ROC curve and area under the precision-recall curve for the
binary link-prediction decision. These are the primary LinkTask metrics.

### Final Accuracy / Final F1

Binary classification accuracy and F1 at the default decision threshold. They are
kept for matrix compatibility with the classification report; AUROC/AUPRC are the
metrics to compare across prompts.

## Notes

- This report is a representative ProG-V2 edge-task sweep, not an exhaustive run
  over every dataset/backbone combination.
- The public report currently uses GCN. Other backbones are available through the
  model registry but are not included in this merged table.
- Regenerate with `bash scripts/bench_paper_grid.sh --task link --gnn_type GCN`,
  then `python scripts/export_final_matrices.py --task Link --gnn_type GCN`.
- The result files intentionally contain only merged metrics and no raw training
  logs or machine-specific execution metadata.
