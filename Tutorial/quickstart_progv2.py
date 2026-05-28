"""Minimal ProG-V2 end-to-end tutorial.

Run from the repository root:

    python Tutorial/quickstart_progv2.py --device cpu --epochs 1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Device passed to bench.py")
    parser.add_argument("--epochs", type=int, default=1, help="Downstream epochs")
    parser.add_argument("--gnn_type", default="GCN", help="Backbone name")
    args = parser.parse_args()

    missing = [
        module
        for module in ("torch", "torch_geometric", "pandas", "openpyxl")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise SystemExit(
            "Missing required packages for the full tutorial run: "
            + ", ".join(missing)
            + '. Install the project first with: pip install -e ".[dev]"'
        )

    run([sys.executable, "scripts/bootstrap_excel_full.py", "--gnn_type", args.gnn_type])

    common = [
        "--gnn_type",
        args.gnn_type,
        "--shot_num",
        "1",
        "--seed",
        "42",
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--pre_train_model_path",
        "None",
        "--num_iter",
        "1",
    ]

    run(
        [
            sys.executable,
            "bench.py",
            "--pretrain_task",
            "NodeTask",
            "--dataset_name",
            "Cora",
            "--prompt_type",
            "None",
            *common,
        ]
    )
    run(
        [
            sys.executable,
            "bench.py",
            "--pretrain_task",
            "GraphTask",
            "--dataset_name",
            "MUTAG",
            "--prompt_type",
            "None",
            *common,
        ]
    )

    print("ProG-V2 quickstart completed.")


if __name__ == "__main__":
    main()
