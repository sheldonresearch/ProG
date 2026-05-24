"""Merge per-shard Excel results from a multi-shard amlt sweep into one file
per (task_kind, shot, dataset).

Each shard's amlt output directory contains an ``excel/`` subdir that mirrors
the standard ``Experiment/ExcelResults/`` layout::

    excel/{Node,Graph}/{shot}shot/{dataset}/GCN_total_results.xlsx

Each shard's xlsx is populated only for the (prompt, pretrain) cells that
shard was responsible for. Merging is a per-cell union: for each (row, col)
take the non-empty value from any shard. Conflicting non-empty values (same
cell written by two shards with different numbers) print a warning and keep
the first.

Usage
-----

  python scripts/merge_shard_excels.py \\
      --input-root amulet_plan_a_out/plan-a-20260528 \\
      --output-root merged_results/plan-a-20260528
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import pandas as pd


def find_xlsx_files(input_root: pathlib.Path) -> dict[tuple[str, str, str], list[pathlib.Path]]:
    """Walk ``input_root`` and group xlsx files by (task_kind, shot, dataset).

    Returns ``{(task, shot_dir, dataset): [path, ...]}``.
    """
    groups: dict[tuple[str, str, str], list[pathlib.Path]] = defaultdict(list)
    # Layout: <input_root>/<shard_dir>/excel/{Node|Graph}/<shot>shot/<dataset>/GCN_total_results.xlsx
    # We accept any depth of <shard_dir> nesting and look for ``excel/Node`` or ``excel/Graph``
    # anywhere underneath.
    for xlsx in input_root.rglob("GCN_total_results.xlsx"):
        # Walk up to find /excel/ and then identify the task/shot/dataset trio.
        try:
            parts = xlsx.relative_to(input_root).parts
        except ValueError:
            continue
        if "excel" not in parts:
            continue
        idx = parts.index("excel")
        try:
            task = parts[idx + 1]
            shot = parts[idx + 2]
            dataset = parts[idx + 3]
        except IndexError:
            print(f"  WARN: skipping malformed path {xlsx}")
            continue
        if task not in {"Node", "Graph"}:
            continue
        groups[(task, shot, dataset)].append(xlsx)
    return groups


def merge_one(paths: list[pathlib.Path]) -> pd.DataFrame:
    """Union columns of the per-shard xlsx files. Last non-empty wins on conflict."""
    merged: pd.DataFrame | None = None
    conflicts: list[str] = []
    for p in paths:
        df = pd.read_excel(p, index_col=0)
        if merged is None:
            merged = df.copy()
            continue
        # Align rows (should always be the same 3 rows: Final Accuracy / F1 / AUROC).
        merged = merged.reindex(merged.index.union(df.index))
        for col in df.columns:
            if col not in merged.columns:
                merged[col] = pd.NA
            for row in df.index:
                new_val = df.at[row, col]
                if pd.isna(new_val) or new_val == "":
                    continue
                cur_val = (
                    merged.at[row, col] if row in merged.index and col in merged.columns else pd.NA
                )
                if pd.isna(cur_val) or cur_val == "":
                    merged.at[row, col] = new_val
                elif cur_val != new_val:
                    conflicts.append(
                        f"  CONFLICT at [{row}, {col}]: '{cur_val}' (kept) vs '{new_val}' (from {p.name})"
                    )
    for c in conflicts[:20]:
        print(c)
    if len(conflicts) > 20:
        print(f"  ... and {len(conflicts) - 20} more conflicts")
    assert merged is not None
    # Stable column order: put combos in alphabetical order so reviewers can scan.
    merged = merged[sorted(merged.columns)]
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-root",
        required=True,
        type=pathlib.Path,
        help="Root containing per-shard amlt output dirs (each with excel/...).",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        type=pathlib.Path,
        help="Where to write merged xlsx files (mirrors Experiment/ExcelResults/ layout).",
    )
    ap.add_argument(
        "--summary-csv",
        type=pathlib.Path,
        default=None,
        help="Optional: write a flat CSV summary (one row per (dataset, shot, col)).",
    )
    args = ap.parse_args()

    if not args.input_root.exists():
        print(f"ERROR: --input-root does not exist: {args.input_root}", file=sys.stderr)
        return 1

    groups = find_xlsx_files(args.input_root)
    if not groups:
        print(f"ERROR: no GCN_total_results.xlsx found under {args.input_root}", file=sys.stderr)
        return 1

    print(
        f"Found {sum(len(v) for v in groups.values())} xlsx file(s) across "
        f"{len(groups)} (task, shot, dataset) group(s) under {args.input_root}"
    )
    print()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for (task, shot, dataset), paths in sorted(groups.items()):
        out_path = args.output_root / task / shot / dataset / "GCN_total_results.xlsx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"== {task} / {shot} / {dataset}  ({len(paths)} shard file(s)) ==")
        merged = merge_one(paths)
        merged.to_excel(out_path)
        n_cells = int(merged.notna().sum().sum())
        n_cols = merged.shape[1]
        print(f"   -> wrote {out_path}  ({n_cols} cols, {n_cells} populated cells)")
        if args.summary_csv is not None:
            for col in merged.columns:
                row = {"task": task, "shot": shot, "dataset": dataset, "combo": col}
                for metric in merged.index:
                    row[metric] = merged.at[metric, col]
                summary_rows.append(row)
        print()

    if args.summary_csv is not None and summary_rows:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
        print(f"Wrote flat summary CSV ({len(summary_rows)} rows) -> {args.summary_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
