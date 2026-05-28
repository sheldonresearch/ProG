"""Bootstrap empty Excel result templates for the full-grid sweep.

This is the extended template generator used by benchmark scripts:
- Covers ALL 12 NODE_TASKS + 11 GRAPH_TASKS (vs paper's 7+8).
- Creates the 3-row index ("Final Accuracy", "Final F1", "Final AUROC") that
  bench.py expects; bench.py will lazily add columns for any
  "{pretrain}+{prompt}" combo it writes, so we leave the column set empty.
- Idempotent: skips files that already exist.

Used by scripts/bench_full_grid.sh.
"""

import argparse
import os

import pandas as pd

from prompt_graph.defines import GRAPH_TASKS, NODE_TASKS
from prompt_graph.utils import excel_result_dir

SHOT_NUMS = (1, 3, 5)
ROW_INDEX = ("Final Accuracy", "Final F1", "Final AUROC")


def ensure_template(task_kind: str, dataset_name: str, shot_num: int, gnn_type: str) -> str:
    """Create an empty template at the bench.py-expected path. Idempotent."""
    file_name = f"{gnn_type}_total_results.xlsx"
    file_path = os.path.join(str(excel_result_dir(task_kind, shot_num, dataset_name)), file_name)
    if os.path.exists(file_path):
        return file_path
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(index=list(ROW_INDEX)).to_excel(file_path)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn_type", "--gnn-type", default="GCN", help="Backbone result prefix")
    args = parser.parse_args()

    created = 0
    skipped = 0
    for dataset in NODE_TASKS:
        for shot in SHOT_NUMS:
            path = ensure_template("Node", dataset, shot, args.gnn_type)
            if os.path.getsize(path) > 0:
                # already-existed files are not re-created above; classify via mtime heuristic
                pass
            print(f"  Node  {dataset:<14} {shot}-shot -> {path}")
            created += 1
    for dataset in GRAPH_TASKS:
        for shot in SHOT_NUMS:
            path = ensure_template("Graph", dataset, shot, args.gnn_type)
            print(f"  Graph {dataset:<14} {shot}-shot -> {path}")
            created += 1
    print(f"Templates ensured: {created} (skipped already-existing among them: {skipped})")


if __name__ == "__main__":
    main()
