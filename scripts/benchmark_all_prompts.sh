#!/usr/bin/env bash
# Benchmark-style coverage run for all 17 registered prompt strategies.
#
# This is the "full bench.py" complement of tests/test_strategy_new_prompts.py
# (which is the 1-epoch pytest smoke). Each case below invokes bench.py with a
# small dataset (Cora for NodeTask, MUTAG for GraphTask) and writes the metric
# to Experiment/ExcelResults/.
#
# Usage:
#   bash scripts/benchmark_all_prompts.sh            # full run (200 epochs)
#   bash scripts/benchmark_all_prompts.sh --fast     # 50 epochs (CI-friendly)
#   bash scripts/benchmark_all_prompts.sh --tag T    # set log column name
#   bash scripts/benchmark_all_prompts.sh --include-broken  # also run combos
#                                                          # currently broken
#                                                          # (see XFAIL list)
#
# Output:
#   - stdout/err mirrored to scripts/baseline_logs/${TAG}_${STAMP}_all_prompts.log
#   - per-case PASS/FAIL line written to that log (errors do NOT abort sweep)
#   - Excel results in Experiment/ExcelResults/...
#
# Differences from scripts/baseline.sh:
#   - baseline.sh is the 3-case Phase-0 regression set (frozen, must match
#     baseline_metrics.md). DO NOT add to it.
#   - This script is the coverage sweep. It is allowed to grow / shrink as
#     strategies are added or fixed. Failures here block neither CI nor merges.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAST=0
TAG="all-prompts"
INCLUDE_BROKEN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST=1; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --include-broken) INCLUDE_BROKEN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS=50
else
    EPOCHS=200
fi

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}_all_prompts.log"

echo "=== All-prompts benchmark sweep ===" | tee "$LOG_FILE"
echo "Tag:            $TAG"                 | tee -a "$LOG_FILE"
echo "Fast mode:      $FAST (epochs=$EPOCHS)" | tee -a "$LOG_FILE"
echo "Include broken: $INCLUDE_BROKEN"       | tee -a "$LOG_FILE"
echo "Log:            $LOG_FILE"             | tee -a "$LOG_FILE"
echo                                          | tee -a "$LOG_FILE"

echo "--- Bootstrap Excel templates ---" | tee -a "$LOG_FILE"
python create_excel_for_bench.py 2>&1 | tee -a "$LOG_FILE" || true
echo | tee -a "$LOG_FILE"

PASS=()
FAIL=()
SKIP=()

# run_case <label> <task> <dataset> <prompt> [extra args...]
run_case() {
    local label="$1"; shift
    local task="$1"; shift
    local dataset="$1"; shift
    local prompt="$1"; shift

    echo "--- $label ---" | tee -a "$LOG_FILE"
    local cmd=(
        python bench.py
        --pretrain_task "$task"
        --dataset_name  "$dataset"
        --prompt_type   "$prompt"
        --gnn_type      GCN
        --shot_num      5
        --seed          42
        --epochs        "$EPOCHS"
        --num_iter      1
        --device        cpu
        --pre_train_model_path "None"
        "$@"
    )
    echo "+ ${cmd[*]}" | tee -a "$LOG_FILE"
    if "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        PASS+=("$label")
    else
        FAIL+=("$label")
    fi
    echo | tee -a "$LOG_FILE"
}

# skip_case <label> <reason>
skip_case() {
    echo "--- SKIP: $1 ($2) ---" | tee -a "$LOG_FILE"
    SKIP+=("$1: $2")
}

# ============================================================================
# NodeTask + Cora — covers the 11 strategies that exercise the node path.
# ============================================================================
echo "### NodeTask + Cora cases ###" | tee -a "$LOG_FILE"

# Working today
run_case "Node/Cora/None"          NodeTask Cora None
run_case "Node/Cora/GPF"           NodeTask Cora GPF
run_case "Node/Cora/GPF-plus"      NodeTask Cora GPF-plus
run_case "Node/Cora/Gprompt"       NodeTask Cora Gprompt
run_case "Node/Cora/All-in-one"    NodeTask Cora All-in-one
run_case "Node/Cora/GPPT"          NodeTask Cora GPPT
if [[ -n "${MULTIGPROMPT_PRETRAIN_PATH:-}" && -f "${MULTIGPROMPT_PRETRAIN_PATH}" ]]; then
    run_case "Node/Cora/MultiGprompt"  NodeTask Cora MultiGprompt \
        --pre_train_model_path "$MULTIGPROMPT_PRETRAIN_PATH"
else
    skip_case "Node/Cora/MultiGprompt" "needs pretrained checkpoint — set MULTIGPROMPT_PRETRAIN_PATH=/path/to/.pth"
fi
run_case "Node/Cora/Prodigy"       NodeTask Cora Prodigy
run_case "Node/Cora/UniPrompt"     NodeTask Cora UniPrompt
run_case "Node/Cora/SelfPro"       NodeTask Cora SelfPro
run_case "Node/Cora/ProNoG"        NodeTask Cora ProNoG
run_case "Node/Cora/PSP"           NodeTask Cora PSP
run_case "Node/Cora/GraphPrompter" NodeTask Cora GraphPrompter
run_case "Node/Cora/EdgePrompt"        NodeTask Cora EdgePrompt
run_case "Node/Cora/EdgePromptplus"    NodeTask Cora EdgePromptplus

# RELIEF is functional but slow (~7 min/epoch on Cora due to O(num_nodes)
# attach_prompt roll-out). Opt-in via INCLUDE_BROKEN to keep --fast runs short.
if [[ "$INCLUDE_BROKEN" -eq 1 ]]; then
    run_case "Node/Cora/RELIEF"        NodeTask Cora RELIEF
else
    skip_case "Node/Cora/RELIEF"       "functional but slow (~7 min/epoch on Cora); use --include-broken to run"
fi

# ============================================================================
# GraphTask + MUTAG — covers the 11 strategies on the graph path.
# ============================================================================
echo "### GraphTask + MUTAG cases ###" | tee -a "$LOG_FILE"

# Working today
run_case "Graph/MUTAG/None"            GraphTask MUTAG None
run_case "Graph/MUTAG/GPF"             GraphTask MUTAG GPF
run_case "Graph/MUTAG/GPF-plus"        GraphTask MUTAG GPF-plus
run_case "Graph/MUTAG/Gprompt"         GraphTask MUTAG Gprompt
run_case "Graph/MUTAG/All-in-one"      GraphTask MUTAG All-in-one
run_case "Graph/MUTAG/GPPT"            GraphTask MUTAG GPPT
run_case "Graph/MUTAG/Prodigy"         GraphTask MUTAG Prodigy
run_case "Graph/MUTAG/DAGPrompT"       GraphTask MUTAG DAGPrompT
run_case "Graph/MUTAG/GraphPrompter"   GraphTask MUTAG GraphPrompter
run_case "Graph/MUTAG/EdgePrompt"      GraphTask MUTAG EdgePrompt
run_case "Graph/MUTAG/EdgePromptplus"  GraphTask MUTAG EdgePromptplus
# RELIEF GraphTask path was wired up in 2026-05-22 bugfix batch-3 (P2.1).
# Small molecules (~17 nodes avg) so no step cap needed.
run_case "Graph/MUTAG/RELIEF"          GraphTask MUTAG RELIEF

# ============================================================================
# Summary
# ============================================================================
echo "=== Summary ===" | tee -a "$LOG_FILE"
printf 'PASS (%d):\n' "${#PASS[@]}" | tee -a "$LOG_FILE"
for c in "${PASS[@]+"${PASS[@]}"}"; do echo "  + $c" | tee -a "$LOG_FILE"; done
printf 'FAIL (%d):\n' "${#FAIL[@]}" | tee -a "$LOG_FILE"
for c in "${FAIL[@]+"${FAIL[@]}"}"; do echo "  - $c" | tee -a "$LOG_FILE"; done
printf 'SKIP (%d):\n' "${#SKIP[@]}" | tee -a "$LOG_FILE"
for c in "${SKIP[@]+"${SKIP[@]}"}"; do echo "  ~ $c" | tee -a "$LOG_FILE"; done

if [[ "${#FAIL[@]}" -gt 0 ]]; then
    exit 1
fi
