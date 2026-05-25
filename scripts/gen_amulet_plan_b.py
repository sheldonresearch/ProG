#!/usr/bin/env python3
"""Generate amulet_plan_b.yaml — 32 single-A100 shards on Feeds covering Plan B.

Plan B target (chat with user 2026-05-24):
  - Datasets: Cora Wisconsin (node) + MUTAG PROTEINS (graph)
  - Pretrains: None + DGI + GraphCL + GraphMAE + Edgepred_Gprompt + MultiGprompt (6)
  - Prompts: all 17 strategies (RELIEF/MultiGprompt may partial-fail per Plan A)
  - Shots: [1, 5]
  - Bench --fast (50 epochs), Pretrain --fast (200 epochs)
  - Total ~512 bench cells across 32 shards

Sharding strategy (32 shards):
  - "Both Node & Graph" prompts (GPF, GPF-plus, Gprompt, GPPT, Prodigy,
    GraphPrompter, EdgePrompt, EdgePromptplus, All-in-one) get one shard for
    Node datasets and one for Graph datasets — keeps each shard a single
    bench_full_grid sweep over 2 datasets × 2 shots × 6 pretrains.
  - All-in-one is heavier; split per (task-type × dataset) so 4 shards total.
  - Node-only prompts (UniPrompt, SelfPro, ProNoG, PSP) pair up: 2 shards.
  - Graph-only DAGPrompT: 1 shard.
  - baseline-mix (None + MultiGprompt) per task type: 2 shards.
  - RELIEF: split per (task-type × dataset) with --num-iter 1 + tight epochs.

Writes to <repo_root>/amulet_plan_b.yaml.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "amulet_plan_b.yaml"

UAI = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity"
    "/userAssignedIdentities/rankfun_aml"
)

NODE_DATASETS = "Cora Wisconsin"
GRAPH_DATASETS = "MUTAG PROTEINS"
NODE_PRETRAINS = "None DGI GraphCL GraphMAE Edgepred_Gprompt MultiGprompt"
GRAPH_PRETRAINS = NODE_PRETRAINS

PRETRAIN_METHODS_NODE = "DGI GraphCL GraphMAE Edgepred_Gprompt MultiGprompt"
PRETRAIN_METHODS_GRAPH = PRETRAIN_METHODS_NODE

DEFAULT_SHOTS = "1 5"

# ---- shard table -----------------------------------------------------------
# Each row: (shard_name, prompts, datasets, pretrain_methods, bench_pretrains,
#            bench_shots, extra_flags)
SHARDS: list[tuple[str, str, str, str, str, str, str]] = [
    # ===== Baselines (None + MultiGprompt) =====
    (
        "baseline-node",
        "None MultiGprompt",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "baseline-graph",
        "None MultiGprompt DAGPrompT",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    # ===== Standard dual-task prompts — Node side =====
    ("gpf-node", "GPF", NODE_DATASETS, PRETRAIN_METHODS_NODE, NODE_PRETRAINS, DEFAULT_SHOTS, ""),
    (
        "gpf-plus-node",
        "GPF-plus",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "gprompt-node",
        "Gprompt",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    ("gppt-node", "GPPT", NODE_DATASETS, PRETRAIN_METHODS_NODE, NODE_PRETRAINS, DEFAULT_SHOTS, ""),
    (
        "prodigy-node",
        "Prodigy",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "graphprompter-node",
        "GraphPrompter",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "edgeprompt-node",
        "EdgePrompt",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "edgepromptplus-node",
        "EdgePromptplus",
        NODE_DATASETS,
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    # ===== Standard dual-task prompts — Graph side =====
    (
        "gpf-graph",
        "GPF",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "gpf-plus-graph",
        "GPF-plus",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "gprompt-graph",
        "Gprompt",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "gppt-graph",
        "GPPT",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "prodigy-graph",
        "Prodigy",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "graphprompter-graph",
        "GraphPrompter",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "edgeprompt-graph",
        "EdgePrompt",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "edgepromptplus-graph",
        "EdgePromptplus",
        GRAPH_DATASETS,
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    # ===== All-in-one (heavy) — one shard per dataset =====
    (
        "all-in-one-cora",
        "All-in-one",
        "Cora",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "all-in-one-wisconsin",
        "All-in-one",
        "Wisconsin",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "all-in-one-mutag",
        "All-in-one",
        "MUTAG",
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "all-in-one-proteins",
        "All-in-one",
        "PROTEINS",
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    # ===== Node-only specialty prompts (paired 2 per shard) =====
    (
        "uni-self-cora",
        "UniPrompt SelfPro",
        "Cora",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "uni-self-wisconsin",
        "UniPrompt SelfPro",
        "Wisconsin",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "pronog-psp-cora",
        "ProNoG PSP",
        "Cora",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    (
        "pronog-psp-wisconsin",
        "ProNoG PSP",
        "Wisconsin",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        DEFAULT_SHOTS,
        "",
    ),
    # ===== RELIEF — split per (task × dataset) with tight budget =====
    # NOTE: RELIEF is very slow on Node (~76s/epoch from Plan A); --num-iter 1
    # caps to 1 trial per cell. We still time out on Wisconsin probably;
    # accept partial results. Graph side (MUTAG/PROTEINS) is much cheaper.
    (
        "relief-cora",
        "RELIEF",
        "Cora",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        "1",
        "--include-slow --num-iter 1",
    ),
    (
        "relief-wisconsin",
        "RELIEF",
        "Wisconsin",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        "1",
        "--include-slow --num-iter 1",
    ),
    (
        "relief-mutag",
        "RELIEF",
        "MUTAG",
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "--include-slow --num-iter 1",
    ),
    (
        "relief-proteins",
        "RELIEF",
        "PROTEINS",
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        DEFAULT_SHOTS,
        "--include-slow --num-iter 1",
    ),
    # ===== 5-shot All-in-one shards for variance (spare capacity) =====
    # These overlap with all-in-one-* above (--shots "1 5"). To avoid double-
    # work, instead use spare slots to re-run the lightest prompts with
    # num-iter 20 for tighter variance estimates on Cora.
    (
        "variance-cora",
        "GPF Gprompt GPPT EdgePromptplus",
        "Cora",
        PRETRAIN_METHODS_NODE,
        NODE_PRETRAINS,
        "1",
        "--num-iter 20",
    ),
    (
        "variance-mutag",
        "GPF Gprompt All-in-one EdgePromptplus",
        "MUTAG",
        PRETRAIN_METHODS_GRAPH,
        GRAPH_PRETRAINS,
        "1",
        "--num-iter 20",
    ),
]


assert len(SHARDS) == 32, f"expected 32 shards, got {len(SHARDS)}"


def render_job(
    shard_name: str,
    prompts: str,
    datasets: str,
    pretrain_methods: str,
    bench_pretrains: str,
    bench_shots: str,
    extra: str,
) -> str:
    """Render one job yaml block: 1×A100 on Feeds-Singularity."""
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
      - export PRETRAIN_METHODS='{pretrain_methods}'
      - export PRETRAIN_DATASETS='{datasets}'
      - export BENCH_PROMPTS='{prompts}'
      - export BENCH_DATASETS='{datasets}'
      - export BENCH_PRETRAINS='{bench_pretrains}'
      - export BENCH_SHOTS='{bench_shots}'
      - export BENCH_EXTRA_FLAGS='{extra}'
      - bash $$AMLT_CODE_DIR/scripts/amlt_shard_run.sh
"""
    return block


def main() -> None:
    header = textwrap.dedent("""\
        # Auto-generated by scripts/gen_amulet_plan_b.py — DO NOT EDIT BY HAND.
        # Plan B sweep on Feeds: 32 × single-A100 shards.
        #   - Datasets: Cora Wisconsin (node) + MUTAG PROTEINS (graph)
        #   - Pretrains: None + DGI + GraphCL + GraphMAE + Edgepred_Gprompt + MultiGprompt
        #   - Prompts: all 17 strategies
        #   - Shots: 1 5
        #
        # Submit:
        #   amlt run amulet_plan_b.yaml plan-b-<date> -d "ProG Plan B on Feeds" -y
        # Monitor:
        #   amlt status plan-b-<date> -v
        # Download all 32 shards' outputs:
        #   amlt results download plan-b-<date> -o amulet_plan_b_out/

        description: ProG Plan B — 17 prompts × {Cora,Wisconsin,MUTAG,PROTEINS} × {1,5}-shot × 6 pretrains on Feeds A100

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
        print(f"  - {s[0]:<24s} prompts=[{s[1]}] datasets=[{s[2]}] shots=[{s[5]}] extra=[{s[6]}]")


if __name__ == "__main__":
    main()
