#!/usr/bin/env python3
"""Generate amulet_plan_b_3shot.yaml — 27 single-A100 shards on Feeds.

Plan B 3-shot follow-up (chat with user 2026-05-24):
  - Same 4 datasets (Cora Wisconsin MUTAG PROTEINS), 6 pretrains, 17 prompts
  - But ONLY 3-shot (--shots "3"), so we get the {1,3,5}-shot triplet
    combined with Plan B's 1-shot + 5-shot results.
  - 27 shards (= 32 quota - 5 still running on plan-b-20260524).

Sharding mirrors Plan B except:
  - DROP variance-* (2 shards): only meaningful as a 1-shot tighter-std re-run
  - DROP relief-cora, relief-wisconsin (2 shards): node RELIEF too slow on
    1 A100 (≥1 h/dataset at 1-shot in Plan A/B); not worth re-running for
    3-shot. Keep relief-mutag (fast on graph) but DROP relief-proteins
    (had ck-pretrain bottleneck in Plan B).
  - So drop 5 shards total → 27 fits in remaining quota.

Writes to <repo_root>/amulet_plan_b_3shot.yaml.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "amulet_plan_b_3shot.yaml"

UAI = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity"
    "/userAssignedIdentities/rankfun_aml"
)

NODE_DATASETS = "Cora Wisconsin"
GRAPH_DATASETS = "MUTAG PROTEINS"
ALL_PRETRAINS = "None DGI GraphCL GraphMAE Edgepred_Gprompt MultiGprompt"
PRETRAIN_METHODS = "DGI GraphCL GraphMAE Edgepred_Gprompt MultiGprompt"

SHOTS = "3"

# Each row: (shard_name, prompts, datasets, pretrain_methods, bench_pretrains,
#            bench_shots, extra_flags)
SHARDS: list[tuple[str, str, str, str, str, str, str]] = [
    # ===== Baselines =====
    (
        "baseline-node-3s",
        "None MultiGprompt",
        NODE_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    (
        "baseline-graph-3s",
        "None MultiGprompt DAGPrompT",
        GRAPH_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    # ===== Standard dual-task prompts — Node =====
    ("gpf-node-3s", "GPF", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gpf-plus-node-3s", "GPF-plus", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gprompt-node-3s", "Gprompt", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gppt-node-3s", "GPPT", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("prodigy-node-3s", "Prodigy", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "graphprompter-node-3s",
        "GraphPrompter",
        NODE_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    ("edgeprompt-node-3s", "EdgePrompt", NODE_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "edgepromptplus-node-3s",
        "EdgePromptplus",
        NODE_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    # ===== Standard dual-task prompts — Graph =====
    ("gpf-graph-3s", "GPF", GRAPH_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gpf-plus-graph-3s", "GPF-plus", GRAPH_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gprompt-graph-3s", "Gprompt", GRAPH_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("gppt-graph-3s", "GPPT", GRAPH_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    ("prodigy-graph-3s", "Prodigy", GRAPH_DATASETS, PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "graphprompter-graph-3s",
        "GraphPrompter",
        GRAPH_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    (
        "edgeprompt-graph-3s",
        "EdgePrompt",
        GRAPH_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    (
        "edgepromptplus-graph-3s",
        "EdgePromptplus",
        GRAPH_DATASETS,
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    # ===== All-in-one (heavy) — one shard per dataset =====
    ("all-in-one-cora-3s", "All-in-one", "Cora", PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "all-in-one-wisconsin-3s",
        "All-in-one",
        "Wisconsin",
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    ("all-in-one-mutag-3s", "All-in-one", "MUTAG", PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "all-in-one-proteins-3s",
        "All-in-one",
        "PROTEINS",
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    # ===== Node-only specialty prompts (paired 2 per shard) =====
    ("uni-self-cora-3s", "UniPrompt SelfPro", "Cora", PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "uni-self-wisconsin-3s",
        "UniPrompt SelfPro",
        "Wisconsin",
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    ("pronog-psp-cora-3s", "ProNoG PSP", "Cora", PRETRAIN_METHODS, ALL_PRETRAINS, SHOTS, ""),
    (
        "pronog-psp-wisconsin-3s",
        "ProNoG PSP",
        "Wisconsin",
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "",
    ),
    # ===== RELIEF — graph only (MUTAG fast; skip slow Node & PROTEINS) =====
    (
        "relief-mutag-3s",
        "RELIEF",
        "MUTAG",
        PRETRAIN_METHODS,
        ALL_PRETRAINS,
        SHOTS,
        "--include-slow --num-iter 1",
    ),
]


assert len(SHARDS) == 27, f"expected 27 shards, got {len(SHARDS)}"


def render_job(
    shard_name: str,
    prompts: str,
    datasets: str,
    pretrain_methods: str,
    bench_pretrains: str,
    bench_shots: str,
    extra: str,
) -> str:
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
        # Auto-generated by scripts/gen_amulet_plan_b_3shot.py — DO NOT EDIT BY HAND.
        # Plan B 3-shot follow-up: 27 × single-A100 shards on Feeds.
        #   - Datasets: Cora Wisconsin (node) + MUTAG PROTEINS (graph)
        #   - Pretrains: None + DGI + GraphCL + GraphMAE + Edgepred_Gprompt + MultiGprompt
        #   - Prompts: 17 strategies (RELIEF only on MUTAG)
        #   - Shots: 3 (combines with Plan B's 1+5 to give {1,3,5}-shot triplet)
        #
        # Submit:
        #   amlt run amulet_plan_b_3shot.yaml plan-b-3shot-<date> -d "..." -y
        # Monitor:
        #   amlt status plan-b-3shot-<date> -v

        description: ProG Plan B 3-shot follow-up — 17 prompts × 4 datasets × 6 pretrains on Feeds A100

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
        print(f"  - {s[0]:<26s} prompts=[{s[1]}] datasets=[{s[2]}] shots=[{s[5]}] extra=[{s[6]}]")


if __name__ == "__main__":
    main()
