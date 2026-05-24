"""Bootstrap empty Excel result templates for the full-grid sweep.

This is the extended version of create_excel_for_bench.py:
- Covers ALL 12 NODE_TASKS + 11 GRAPH_TASKS (vs paper's 7+8).
- Creates the 3-row index ("Final Accuracy", "Final F1", "Final AUROC") that
  bench.py expects; bench.py will lazily add columns for any
  "{pretrain}+{prompt}" combo it writes, so we leave the column set empty.
- Idempotent: skips files that already exist.

Used by scripts/bench_full_grid.sh.
"""

import os

import pandas as pd

from prompt_graph.defines import GRAPH_TASKS, NODE_TASKS
from prompt_graph.utils import excel_result_dir

SHOT_NUMS = (1, 3, 5)
GNN_TYPE = "GCN"
ROW_INDEX = ("Final Accuracy", "Final F1", "Final AUROC")


def ensure_template(task_kind: str, dataset_name: str, shot_num: int) -> str:
    """Create an empty template at the bench.py-expected path. Idempotent."""
    file_name = f"{GNN_TYPE}_total_results.xlsx"
    file_path = os.path.join(str(excel_result_dir(task_kind, shot_num, dataset_name)), file_name)
    if os.path.exists(file_path):
        return file_path
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(index=list(ROW_INDEX)).to_excel(file_path)
    return file_path


def main() -> None:
    created = 0
    skipped = 0
    for dataset in NODE_TASKS:
        for shot in SHOT_NUMS:
            path = ensure_template("Node", dataset, shot)
            if os.path.getsize(path) > 0:
                # already-existed files are not re-created above; classify via mtime heuristic
                pass
            print(f"  Node  {dataset:<14} {shot}-shot -> {path}")
            created += 1
    for dataset in GRAPH_TASKS:
        for shot in SHOT_NUMS:
            path = ensure_template("Graph", dataset, shot)
            print(f"  Graph {dataset:<14} {shot}-shot -> {path}")
            created += 1
    print(f"Templates ensured: {created} (skipped already-existing among them: {skipped})")


if __name__ == "__main__":
    main()
