"""Generate a Cora-only follow-up amlt yaml that fills in the NodeTask cells
that failed in plan-a-20260528 due to the induced_graph CUDA/CPU device
mismatch bug (now fixed in prompt_graph/data/induced_graph.py).

Skips:
- MUTAG-only shards (all-in-one-mutag) — already done in v5
- RELIEF (too slow at 76s/epoch on Cora; not worth a follow-up)
- The MUTAG arm of dual-dataset shards (we already have those cells)

Run after a code change to regenerate:
    python scripts/gen_amulet_plan_a_cora_fixup.py > amulet_plan_a_cora_fixup.yaml
"""

from __future__ import annotations

UAI = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/"
    "AMLStudio/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

# (shard_name, BENCH_PROMPTS, BENCH_PRETRAINS, EXTRA_FLAGS)
# All shards run only on Cora (NodeTask).
SHARDS: list[tuple[str, str, str, str]] = [
    ("baseline-mix-cora", "None MultiGprompt", "DGI GraphCL GraphMAE MultiGprompt", ""),
    ("gpf-cora", "GPF", "DGI GraphCL GraphMAE", ""),
    ("gpf-plus-cora", "GPF-plus", "DGI GraphCL GraphMAE", ""),
    ("gprompt-cora", "Gprompt", "DGI GraphCL GraphMAE", ""),
    ("all-in-one-cora-v2", "All-in-one", "DGI GraphCL GraphMAE", ""),
    ("gppt-cora", "GPPT", "DGI GraphCL GraphMAE", ""),
    ("prodigy-cora", "Prodigy", "DGI GraphCL GraphMAE", ""),
    ("graphprompter-cora", "GraphPrompter", "DGI GraphCL GraphMAE", ""),
    ("edgeprompt-cora", "EdgePrompt", "DGI GraphCL GraphMAE", ""),
    ("edgepromptplus-cora", "EdgePromptplus", "DGI GraphCL GraphMAE", ""),
    ("uni-self-cora", "UniPrompt SelfPro", "DGI GraphCL GraphMAE", ""),
    ("pronog-psp-cora", "ProNoG PSP", "DGI GraphCL GraphMAE", ""),
    (
        "variance-boost-cora",
        "GPF Gprompt All-in-one EdgePromptplus",
        "DGI GraphCL GraphMAE",
        "--num-iter 20",
    ),
]


def render_job(shard_name: str, prompts: str, methods: str, extra: str) -> str:
    # methods is the BENCH_PRETRAINS-style list (may include 'None').
    # PRETRAIN_METHODS skips 'None' (no checkpoint needed for the baseline).
    pretrain_methods = " ".join(m for m in methods.split() if m != "None")
    # BENCH_PRETRAINS: always test 'None' baseline + everything we pretrained.
    bench_pretrains = ("None " + pretrain_methods).strip()
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
      - export PRETRAIN_DATASETS='Cora'
      - export BENCH_PROMPTS='{prompts}'
      - export BENCH_DATASETS='Cora'
      - export BENCH_PRETRAINS='{bench_pretrains}'
      - export BENCH_SHOTS='1'
      - export BENCH_EXTRA_FLAGS='{extra}'
      - bash scripts/amlt_shard_run.sh
"""
    return block


def main() -> None:
    print("description: Plan A Cora follow-up — re-run NodeTask cells after induced_graph CUDA fix")
    print("")
    print("environment:")
    print("  image: amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04")
    print("  setup:")
    print("    - python --version")
    print("")
    print("code:")
    print("  local_dir: $CONFIG_DIR")
    print("")
    print("target:")
    print("  service: sing")
    print("  name: Feeds")
    print("  workspace_name: CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity")
    print("")
    print("jobs:")
    for shard in SHARDS:
        print(render_job(*shard), end="")


if __name__ == "__main__":
    main()
