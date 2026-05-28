#!/usr/bin/env bash
# Full project-grid sweep — all strategies × all datasets × all pretrains × all
# shots that this repo can actually run, with paper-style 3 metrics
# (Accuracy / F1 / AUROC) written to Experiment/ExcelResults/.
#
# Coverage:
#   - 17 strategies (per STRATEGY_REGISTRY)
#       Both Node+Graph (10): None, GPF, GPF-plus, Gprompt, All-in-one, GPPT,
#                             Prodigy, GraphPrompter, EdgePrompt, EdgePromptplus
#       Both with slow gate (1): RELIEF (slow on NodeTask; opt-in via --include-slow)
#       Node-only (5): MultiGprompt, UniPrompt, SelfPro, ProNoG, PSP
#       Graph-only (1): DAGPrompT
#   - 23 datasets (NODE_TASKS 12 + GRAPH_TASKS 11)
#   - 7 pretrain variants: None (scratch), DGI, GraphMAE, Edgepred_GPPT,
#                          Edgepred_Gprompt, GraphCL, SimGRACE
#       (MultiGprompt strategy is paired ONLY with the MultiGprompt pretrain.)
#   - 3 shots (1, 3, 5)
#
# Prerequisites:
#   1. bash scripts/pretrain_full_grid.sh    # produces all .pth files
#   2. python scripts/bootstrap_excel_full.py  # auto-run by this script
#
# Per-cell rules:
#   - prompt == None             → only pretrain == None (supervised baseline)
#   - prompt == MultiGprompt     → only pretrain == MultiGprompt
#   - other prompts              → all 7 pretrain variants
#   - Node-only / Graph-only strategies are filtered by --task
#   - RELIEF on NodeTask is skipped unless --include-slow is set
#
# Usage:
#   bash scripts/bench_full_grid.sh                              # full sweep
#   bash scripts/bench_full_grid.sh --fast                       # 50 epochs
#   bash scripts/bench_full_grid.sh --shots "1"                  # 1-shot only
#   bash scripts/bench_full_grid.sh --task node                  # node side only
#   bash scripts/bench_full_grid.sh --datasets "Cora MUTAG"
#   bash scripts/bench_full_grid.sh --prompts "GPF GPF-plus All-in-one"
#   bash scripts/bench_full_grid.sh --pretrains "DGI GraphCL"
#   bash scripts/bench_full_grid.sh --device 0                   # GPU 0
#   bash scripts/bench_full_grid.sh --gnn_type GAT               # backbone
#   bash scripts/bench_full_grid.sh --include-slow               # enable RELIEF/NodeTask
#   bash scripts/bench_full_grid.sh --exclude-ogb                # skip ogbn-arxiv + ogbg-ppa
#   bash scripts/bench_full_grid.sh --allow-missing              # train from scratch when .pth missing
#
# Output:
#   - Excel: Experiment/ExcelResults/<Node|Graph>/<shot>shot/<dataset>/<gnn_type>_total_results.xlsx
#       columns appended as "{pretrain}+{prompt}"
#       rows: Final Accuracy, Final F1, Final AUROC
#   - Log:   scripts/baseline_logs/${TAG}_${STAMP}_bench_full.log
#
# Cost note: full sweep is ~10K bench cells. Use filters to slice.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- project grid -----------------------------------------------------------
ALL_NODE_DATASETS=(Cora CiteSeer PubMed Wisconsin Texas Actor Computers Photo Reddit WikiCS Flickr ogbn-arxiv)
ALL_GRAPH_DATASETS=(MUTAG ENZYMES COLLAB PROTEINS IMDB-BINARY REDDIT-BINARY COX2 BZR PTC_MR DD ogbg-ppa)
ALL_PRETRAINS=(None DGI GraphMAE Edgepred_GPPT Edgepred_Gprompt GraphCL SimGRACE MultiGprompt)
ALL_PROMPTS=(None GPF GPF-plus Gprompt All-in-one GPPT Prodigy GraphPrompter EdgePrompt EdgePromptplus RELIEF MultiGprompt UniPrompt SelfPro ProNoG PSP DAGPrompT)
DEFAULT_SHOTS=(1 3 5)
OGB_DATASETS=(ogbn-arxiv ogbg-ppa)

# Per-strategy task applicability (see verify-strategy-tasks report).
# Bash 3.x compatible (no associative arrays).
NODE_OK_PROMPTS=(None GPF GPF-plus Gprompt All-in-one GPPT Prodigy GraphPrompter EdgePrompt EdgePromptplus MultiGprompt UniPrompt SelfPro ProNoG PSP RELIEF)
GRAPH_OK_PROMPTS=(None GPF GPF-plus Gprompt All-in-one GPPT Prodigy GraphPrompter EdgePrompt EdgePromptplus DAGPrompT RELIEF)
# RELIEF on NodeTask is functional but slow; gated by --include-slow.
SLOW_NODE_PROMPTS=(RELIEF)

# ---- defaults ---------------------------------------------------------------
TASK="all"
EPOCHS=200
DEVICE="cpu"
GNN_TYPE="GCN"
TAG="bench-full"
FAST=0
ALLOW_MISSING=0
INCLUDE_SLOW=0
EXCLUDE_OGB=0
SHOTS=()
DATASETS=()
PROMPTS=()
PRETRAINS=()
NUM_ITER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --gnn_type|--gnn-type) GNN_TYPE="$2"; shift 2 ;;
        --tag)           TAG="$2"; shift 2 ;;
        --fast)          FAST=1; shift ;;
        --shots)         read -ra SHOTS     <<< "$2"; shift 2 ;;
        --datasets)      read -ra DATASETS  <<< "$2"; shift 2 ;;
        --prompts)       read -ra PROMPTS   <<< "$2"; shift 2 ;;
        --pretrains)     read -ra PRETRAINS <<< "$2"; shift 2 ;;
        --num-iter)      NUM_ITER="$2"; shift 2 ;;
        --include-slow)  INCLUDE_SLOW=1; shift ;;
        --exclude-ogb)   EXCLUDE_OGB=1; shift ;;
        --allow-missing) ALLOW_MISSING=1; shift ;;
        -h|--help)       sed -n '2,55p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS=50
fi

if [[ "${#SHOTS[@]}"     -eq 0 ]]; then SHOTS=("${DEFAULT_SHOTS[@]}");        fi
if [[ "${#PROMPTS[@]}"   -eq 0 ]]; then PROMPTS=("${ALL_PROMPTS[@]}");        fi
if [[ "${#PRETRAINS[@]}" -eq 0 ]]; then PRETRAINS=("${ALL_PRETRAINS[@]}");    fi

contains() {
    local needle="$1"; shift
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

is_ogb() {
    contains "$1" "${OGB_DATASETS[@]}"
}

ckpt_path() {
    local method="$1" dataset="$2"
    if [[ "$method" == "MultiGprompt" ]]; then
        echo "$REPO_ROOT/Experiment/pre_trained_model/${dataset}/MultiGprompt.pth"
    else
        echo "$REPO_ROOT/Experiment/pre_trained_model/${dataset}/${method}.${GNN_TYPE}.128hidden_dim.pth"
    fi
}

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}_bench_full.log"

echo "=== Full bench grid sweep ==="                | tee "$LOG_FILE"
echo "Tag:           $TAG"                           | tee -a "$LOG_FILE"
echo "Task:          $TASK"                          | tee -a "$LOG_FILE"
echo "Shots:         ${SHOTS[*]}"                    | tee -a "$LOG_FILE"
echo "Prompts:       ${PROMPTS[*]}"                  | tee -a "$LOG_FILE"
echo "Pretrains:     ${PRETRAINS[*]}"                | tee -a "$LOG_FILE"
echo "Epochs:        $EPOCHS  (fast=$FAST)"          | tee -a "$LOG_FILE"
echo "Device:        $DEVICE"                        | tee -a "$LOG_FILE"
echo "GNN type:      $GNN_TYPE"                      | tee -a "$LOG_FILE"
echo "Num iter:      ${NUM_ITER:-bench-default}"     | tee -a "$LOG_FILE"
echo "Include slow:  $INCLUDE_SLOW"                  | tee -a "$LOG_FILE"
echo "Exclude OGB:   $EXCLUDE_OGB"                   | tee -a "$LOG_FILE"
echo "Allow missing: $ALLOW_MISSING"                 | tee -a "$LOG_FILE"
echo "Log:           $LOG_FILE"                      | tee -a "$LOG_FILE"
echo                                                 | tee -a "$LOG_FILE"

echo "--- Bootstrap Excel templates (all 23 datasets, 3 shots) ---" | tee -a "$LOG_FILE"
python scripts/bootstrap_excel_full.py --gnn_type "$GNN_TYPE" 2>&1 | tee -a "$LOG_FILE" || true
echo | tee -a "$LOG_FILE"

# Build (task_kind, dataset) cases ------------------------------------------
declare -a CASES
add_cases() {
    local kind="$1"; shift
    local ds_list=("$@")
    if [[ "${#DATASETS[@]}" -gt 0 ]]; then
        for d in "${ds_list[@]}"; do
            if contains "$d" "${DATASETS[@]}"; then
                CASES+=("${kind}|${d}")
            fi
        done
    else
        for d in "${ds_list[@]}"; do
            CASES+=("${kind}|${d}")
        done
    fi
}
case "$TASK" in
    node)  add_cases NodeTask  "${ALL_NODE_DATASETS[@]}"  ;;
    graph) add_cases GraphTask "${ALL_GRAPH_DATASETS[@]}" ;;
    all)
        add_cases NodeTask  "${ALL_NODE_DATASETS[@]}"
        add_cases GraphTask "${ALL_GRAPH_DATASETS[@]}"
        ;;
    *) echo "--task must be node|graph|all" >&2; exit 1 ;;
esac

PASS=()
FAIL=()
SKIP=()

for shot in "${SHOTS[@]}"; do
    for entry in "${CASES[@]}"; do
        IFS='|' read -r task_kind dataset <<< "$entry"

        if [[ "$EXCLUDE_OGB" -eq 1 ]] && is_ogb "$dataset"; then
            label="${task_kind}/${dataset}/${shot}shot"
            echo "--- SKIP (--exclude-ogb): $label ---" | tee -a "$LOG_FILE"
            SKIP+=("$label")
            continue
        fi

        for prompt in "${PROMPTS[@]}"; do
            # Per-strategy task gating
            if [[ "$task_kind" == "NodeTask" ]]; then
                contains "$prompt" "${NODE_OK_PROMPTS[@]}" || continue
                if contains "$prompt" "${SLOW_NODE_PROMPTS[@]}" && [[ "$INCLUDE_SLOW" -ne 1 ]]; then
                    skip_label="${task_kind}/${dataset}/${shot}shot/*+${prompt}"
                    echo "--- SKIP (slow, use --include-slow): $skip_label ---" | tee -a "$LOG_FILE"
                    SKIP+=("$skip_label")
                    continue
                fi
            else
                contains "$prompt" "${GRAPH_OK_PROMPTS[@]}" || continue
            fi

            for pretrain in "${PRETRAINS[@]}"; do
                # Skip non-paper cells:
                # prompt==None pairs only with pretrain==None (supervised baseline)
                if [[ "$prompt" == "None" && "$pretrain" != "None" ]]; then continue; fi
                if [[ "$pretrain" == "None" && "$prompt" != "None" ]] && [[ "$prompt" == "MultiGprompt" ]]; then
                    continue
                fi
                # MultiGprompt strategy needs the MultiGprompt pretrain
                if [[ "$prompt" == "MultiGprompt" && "$pretrain" != "MultiGprompt" ]]; then continue; fi
                if [[ "$prompt" != "MultiGprompt" && "$pretrain" == "MultiGprompt" ]]; then continue; fi

                label="${task_kind}/${dataset}/${shot}shot/${pretrain}+${prompt}"
                ckpt_arg="None"
                if [[ "$pretrain" != "None" ]]; then
                    ckpt="$(ckpt_path "$pretrain" "$dataset")"
                    if [[ -f "$ckpt" ]]; then
                        ckpt_arg="$ckpt"
                    elif [[ "$ALLOW_MISSING" -eq 1 ]]; then
                        echo "--- WARN $label : checkpoint missing, training from scratch ---" \
                            | tee -a "$LOG_FILE"
                        ckpt_arg="None"
                    else
                        echo "--- SKIP $label : missing $ckpt ---" | tee -a "$LOG_FILE"
                        SKIP+=("$label")
                        continue
                    fi
                fi

                echo "--- $label ---" | tee -a "$LOG_FILE"
                cmd=(
                    python bench.py
                    --pretrain_task "$task_kind"
                    --dataset_name  "$dataset"
                    --prompt_type   "$prompt"
                    --gnn_type      "$GNN_TYPE"
                    --shot_num      "$shot"
                    --seed          42
                    --epochs        "$EPOCHS"
                    --device        "$DEVICE"
                    --pre_train_model_path "$ckpt_arg"
                )
                if [[ -n "$NUM_ITER" ]]; then
                    cmd+=(--num_iter "$NUM_ITER")
                fi
                echo "+ ${cmd[*]}" | tee -a "$LOG_FILE"
                if "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
                    PASS+=("$label")
                else
                    FAIL+=("$label")
                fi
                echo | tee -a "$LOG_FILE"
            done
        done
    done
done

echo "=== Summary ===" | tee -a "$LOG_FILE"
printf 'PASS (%d):\n' "${#PASS[@]}" | tee -a "$LOG_FILE"
for c in "${PASS[@]+"${PASS[@]}"}"; do echo "  + $c" | tee -a "$LOG_FILE"; done
printf 'SKIP (%d):\n' "${#SKIP[@]}" | tee -a "$LOG_FILE"
for c in "${SKIP[@]+"${SKIP[@]}"}"; do echo "  ~ $c" | tee -a "$LOG_FILE"; done
printf 'FAIL (%d):\n' "${#FAIL[@]}" | tee -a "$LOG_FILE"
for c in "${FAIL[@]+"${FAIL[@]}"}"; do echo "  - $c" | tee -a "$LOG_FILE"; done

if [[ "${#FAIL[@]}" -gt 0 ]]; then
    exit 1
fi
