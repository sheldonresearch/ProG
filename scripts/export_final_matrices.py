"""Export benchmark Excel matrices into a public report directory.

This script turns the per-dataset files written by ``bench.py`` into the same
shape as ``results/benchmark-gcn/final_matrices.xlsx``:

* copy non-empty ``<gnn_type>_total_results.xlsx`` matrices into
  ``<output-root>/<Task>/<shot>shot/<dataset>/``;
* write ``summary.csv`` with one row per populated ``pretrain+prompt`` combo;
* write ``final_matrices.xlsx`` with one sheet per ``Task_shot_dataset``.

Example:

    python scripts/export_final_matrices.py \\
      --input-root Experiment/ExcelResults \\
      --output-root results/link-prediction-gcn \\
      --task Link \\
      --gnn_type GCN
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _parse_tasks(raw: str) -> set[str]:
    if raw.lower() == "all":
        return {"Node", "Graph", "Link"}
    mapping = {"node": "Node", "graph": "Graph", "link": "Link"}
    tasks = set()
    for item in raw.replace(",", " ").split():
        key = item.lower()
        if key not in mapping:
            raise ValueError(f"Unknown task {item!r}; expected node, graph, link, or all.")
        tasks.add(mapping[key])
    return tasks


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=0)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, [col for col in df.columns if not str(col).startswith("Unnamed:")]]
    return df


def _sheet_name(task: str, shot: str, dataset: str) -> str:
    name = f"{task}_{shot}_{dataset}"
    return name[:31]


def export_report(input_root: Path, output_root: Path, gnn_type: str, tasks: set[str]) -> int:
    result_name = f"{gnn_type}_total_results.xlsx"
    matrices: list[tuple[str, str, str, Path, pd.DataFrame]] = []
    summary_rows: list[dict[str, str]] = []

    for task in sorted(tasks):
        task_root = input_root / task
        if not task_root.exists():
            continue
        for path in sorted(task_root.glob(f"*shot/*/{result_name}")):
            shot = path.parent.parent.name
            dataset = path.parent.name
            df = _read_matrix(path)
            if df.empty or df.notna().sum().sum() == 0:
                continue

            out_path = output_root / task / shot / dataset / result_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != out_path.resolve():
                shutil.copy2(path, out_path)
            matrices.append((task, shot, dataset, out_path, df))

            for combo in df.columns:
                metric_values = {
                    str(metric): df.at[metric, combo]
                    for metric in df.index
                    if pd.notna(df.at[metric, combo])
                }
                if not metric_values:
                    continue
                summary_rows.append(
                    {
                        "task": task,
                        "shot": shot,
                        "dataset": dataset,
                        "combo": str(combo),
                        **metric_values,
                    }
                )

    if not matrices:
        raise FileNotFoundError(
            f"No populated {result_name} files found under {input_root} for tasks {sorted(tasks)}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    workbook_path = output_root / "final_matrices.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        used_names: set[str] = set()
        for task, shot, dataset, _path, df in matrices:
            base = _sheet_name(task, shot, dataset)
            sheet = base
            suffix = 1
            while sheet in used_names:
                suffix += 1
                sheet = f"{base[: 31 - len(str(suffix)) - 1]}_{suffix}"
            used_names.add(sheet)
            df.to_excel(writer, sheet_name=sheet)

    print(f"Copied {len(matrices)} populated matrix file(s) into {output_root}")
    print(f"Wrote {len(summary_rows)} summary row(s) -> {summary_path}")
    print(f"Wrote workbook -> {workbook_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("Experiment/ExcelResults"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gnn_type", "--gnn-type", default="GCN")
    parser.add_argument("--task", default="all", help="node, graph, link, or all")
    args = parser.parse_args()

    tasks = _parse_tasks(args.task)
    return export_report(args.input_root, args.output_root, args.gnn_type, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
