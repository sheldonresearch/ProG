#!/usr/bin/env python3
"""Generate amulet_plan_a.yaml — 16 single-A100 shards on Feeds covering Plan A.

Plan A target (chat with user 2026-05-24):
  - Datasets: Cora (node) + MUTAG (graph)
  - Pretrains: None + DGI + GraphCL + GraphMAE + MultiGprompt (5)
  - Prompts: all 17 strategies in STRATEGY_REGISTRY
  - Shots: [1]
  - Bench --fast (50 epochs), Pretrain --fast (200 epochs)
  - Total ~103 bench cells across 16 shards

Sharding strategy: roughly one prompt per shard, with light prompts merged
and the slow RELIEF prompt split per-dataset with --num-iter 1.

Writes to <repo_root>/amulet_plan_a.yaml.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "amulet_plan_a.yaml"

UAI = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity"
    "/userAssignedIdentities/rankfun_aml"
)

# ---- shard table -----------------------------------------------------------
# Each row: (shard_name, prompts, datasets, pretrain_methods, extra_flags)
#
# pretrain_methods is the *minimal* set this shard needs (script is idempotent
# so over-provisioning is harmless but wastes a few minutes per shard).
# extra_flags is space-separated, passed verbatim to bench_full_grid.sh.
#
# Total: 16 shards = 16 single-A100 jobs.
SHARDS: list[tuple[str, str, str, str, str]] = [
    # 01. Light grab-bag: baseline + MultiGprompt (node-only) + DAGPrompT (graph-only)
    ("baseline-mix",
     "None MultiGprompt DAGPrompT",
     "Cora MUTAG",
     "DGI GraphCL GraphMAE MultiGprompt",
     ""),
    # 02-04. Standard dual-task prompts, both datasets
    ("gpf",            "GPF",            "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    ("gpf-plus",       "GPF-plus",       "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    ("gprompt",        "Gprompt",        "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    # 05-06. All-in-one is heavy — split by dataset
    ("all-in-one-cora",  "All-in-one",   "Cora",       "DGI GraphCL GraphMAE", ""),
    ("all-in-one-mutag", "All-in-one",   "MUTAG",      "DGI GraphCL GraphMAE", ""),
    # 07. GPPT (bench.py auto-forces num_iter=1; fast)
    ("gppt",           "GPPT",           "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    # 08-11. Other dual-task prompts
    ("prodigy",        "Prodigy",        "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    ("graphprompter",  "GraphPrompter",  "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    ("edgeprompt",     "EdgePrompt",     "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    ("edgepromptplus", "EdgePromptplus", "Cora MUTAG", "DGI GraphCL GraphMAE", ""),
    # 12-13. Node-only specialty prompts (paired 2 per shard to balance load)
    ("uni-self",       "UniPrompt SelfPro", "Cora",    "DGI GraphCL GraphMAE", ""),
    ("pronog-psp",     "ProNoG PSP",        "Cora",    "DGI GraphCL GraphMAE", ""),
    # 14. RELIEF on Cora — known slow on Node; --num-iter 1 caps to 1 trial.
    # Note: --fast forces --epochs=50 in bench_full_grid.sh regardless of any
    # explicit --epochs flag (see line 106-108), so we don't bother passing it.
    ("relief-cora",    "RELIEF",         "Cora",       "DGI GraphCL GraphMAE",
     "--include-slow --num-iter 1"),
    # 15. RELIEF on MUTAG — fast on Graph; default --fast budget
    ("relief-mutag",   "RELIEF",         "MUTAG",      "DGI GraphCL GraphMAE",
     "--include-slow --num-iter 1"),
    # 16. Variance-boost: re-run 4 most-impactful prompts with num_iter=20 for
    #     tighter stats on the lightest dataset (Cora). Cheap; spare capacity.
    ("variance-boost",
     "GPF Gprompt All-in-one EdgePromptplus",
     "Cora",
     "DGI GraphCL GraphMAE",
     "--num-iter 20"),
]


def render_job(shard_name: str,
               prompts: str,
               datasets: str,
               methods: str,
               extra: str) -> str:
    """Render one job: yaml block, single A100, Singularity Feeds."""
    # NOTE on $$-escaping (skill §4.9): amlt runs `command:` through
    # string.Template.substitute before shell exec, so any `${VAR}` must be
    # `$${VAR}` and any `$VAR` must be `$$VAR`. We avoid those entirely in
    # this job by exporting our shard config and then exec'ing the runner.
    block = f"""\
  - name: {shard_name}
    sku: 80G1-A100
    sla_tier: Standard
    submit_args:
      env:
        AMLT_DOCKERFILE_TEMPLATE: DEFAULT
        _AZUREML_SINGULARITY_JOB_UAI: {UAI}
    command:
      - set -euo pipefail
      - export SHARD_NAME='{shard_name}'
      - export PRETRAIN_METHODS='{methods}'
      - export PRETRAIN_DATASETS='{datasets}'
      - export BENCH_PROMPTS='{prompts}'
      - export BENCH_DATASETS='{datasets}'
      - export BENCH_PRETRAINS='None DGI GraphCL GraphMAE MultiGprompt'
      - export BENCH_SHOTS='1'
      - export BENCH_EXTRA_FLAGS='{extra}'
      - bash $$AMLT_CODE_DIR/scripts/amlt_shard_run.sh
"""
    return block


def main() -> None:
    header = textwrap.dedent(f"""\
        # Auto-generated by scripts/gen_amulet_plan_a.py — DO NOT EDIT BY HAND.
        # Plan A sweep on Feeds: 16 × single-A100 shards.
        #   - Datasets: Cora (node) + MUTAG (graph)
        #   - Pretrains: None + DGI + GraphCL + GraphMAE + MultiGprompt
        #   - Prompts: all 17 strategies (RELIEF split per-dataset with tight budget)
        #   - Shots: 1
        #
        # Submit:
        #   amlt run amulet_plan_a.yaml plan-a-<date> -d "ProG Plan A smoke on Feeds" -y
        # Monitor:
        #   amlt status plan-a-<date> -v
        # Download all 16 shards' outputs:
        #   amlt results download plan-a-<date> -o amulet_plan_a_out/

        description: ProG Plan A — 17 prompts × Cora+MUTAG × 1-shot on Feeds A100

        environment:
          image: amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04
          setup:
            - python --version
            - nvidia-smi || true

        code:
          local_dir: $CONFIG_DIR

        target:
          service: sing
          name: Feeds
          workspace_name: CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity

        jobs:
        """)

    body = "".join(render_job(*s) for s in SHARDS)
    OUT_PATH.write_text(header + body, encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(SHARDS)} shards")
    for s in SHARDS:
        print(f"  - {s[0]:<20s} prompts=[{s[1]}] datasets=[{s[2]}] extra=[{s[4]}]")


if __name__ == "__main__":
    main()
