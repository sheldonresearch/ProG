#!/usr/bin/env bash
# Full Section 5.1 "Overall Performance" sweep.
#
# Iterates: shot_num × dataset × prompt_type × pretrain_method
#   - shot_num    ∈ {1, 3, 5}          (paper Tables 2/3 + Tables 7-10)
#   - datasets    = 7 node + 8 graph (paper Table 1, 15 total)
#   - prompt_type ∈ {None, GPPT, All-in-one, Gprompt, GPF, GPF-plus}
#   - pretrain    ∈ {None, DGI, GraphMAE, Edgepred_GPPT, Edgepred_Gprompt,
#                    GraphCL, SimGRACE}    (None = train from scratch)
#
# For each cell, calls bench.py once. Results are appended into the Excel files
# at Experiment/ExcelResults/<task>/<shot>shot/<dataset>/GCN_total_results.xlsx
# under the column "{pretrain}+{prompt}".
#
# Prerequisite: run scripts/pretrain_paper_grid.sh FIRST to produce all .pth
# files. This script will SKIP any (pretrain, dataset) combo whose checkpoint
# is missing (unless --allow-missing is set, which falls back to from-scratch).
#
# Usage:
#   bash scripts/bench_overall_performance.sh                         # full
#   bash scripts/bench_overall_performance.sh --fast                  # 50 ep
#   bash scripts/bench_overall_performance.sh --shots "1"             # 1-shot only
#   bash scripts/bench_overall_performance.sh --task node             # node only
#   bash scripts/bench_overall_performance.sh --datasets "Cora MUTAG" # subset
#   bash scripts/bench_overall_performance.sh --prompts "GPF GPF-plus"
#   bash scripts/bench_overall_performance.sh --pretrains "DGI None"
#   bash scripts/bench_overall_performance.sh --device 0              # GPU 0
#   bash scripts/bench_overall_performance.sh --allow-missing         # train from scratch when .pth missing
#
# Output:
#   - Excel: Experiment/ExcelResults/<Node|Graph>/<shot>shot/<dataset>/GCN_total_results.xlsx
#   - Log:   scripts/baseline_logs/${TAG}_${STAMP}_overall_performance.log
#
# Cost note: 3 shots × 15 datasets × 6 prompts × 7 pretrain-variants ≈ 1890
# bench runs (each random-search 10 trials by default). Bring snacks.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- paper grid -------------------------------------------------------------
PAPER_NODE_DATASETS=(Cora CiteSeer PubMed Wisconsin Texas Actor ogbn-arxiv)
PAPER_GRAPH_DATASETS=(MUTAG IMDB-BINARY COLLAB PROTEINS ENZYMES COX2 BZR DD)
PAPER_PRETRAINS=(None DGI GraphMAE Edgepred_GPPT Edgepred_Gprompt GraphCL SimGRACE)
PAPER_PROMPTS=(None GPPT All-in-one Gprompt GPF GPF-plus)
DEFAULT_SHOTS=(1 3 5)

# ---- defaults ---------------------------------------------------------------
TASK="all"
EPOCHS=200
DEVICE="cpu"
TAG="overall-perf"
FAST=0
ALLOW_MISSING=0
SHOTS=()
DATASETS=()
PROMPTS=()
PRETRAINS=()
NUM_ITER=""   # empty -> bench.py's default (10, or 1 for special datasets)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)        TASK="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --fast)        FAST=1; shift ;;
        --shots)       read -ra SHOTS     <<< "$2"; shift 2 ;;
        --datasets)    read -ra DATASETS  <<< "$2"; shift 2 ;;
        --prompts)     read -ra PROMPTS   <<< "$2"; shift 2 ;;
        --pretrains)   read -ra PRETRAINS <<< "$2"; shift 2 ;;
        --num-iter)    NUM_ITER="$2"; shift 2 ;;
        --allow-missing) ALLOW_MISSING=1; shift ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS=50
fi

if [[ "${#SHOTS[@]}"     -eq 0 ]]; then SHOTS=("${DEFAULT_SHOTS[@]}");        fi
if [[ "${#PROMPTS[@]}"   -eq 0 ]]; then PROMPTS=("${PAPER_PROMPTS[@]}");      fi
if [[ "${#PRETRAINS[@]}" -eq 0 ]]; then PRETRAINS=("${PAPER_PRETRAINS[@]}");  fi

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}_overall_performance.log"

echo "=== Overall Performance sweep ==="              | tee "$LOG_FILE"
echo "Tag:        $TAG"                                | tee -a "$LOG_FILE"
echo "Task:       $TASK"                               | tee -a "$LOG_FILE"
echo "Shots:      ${SHOTS[*]}"                         | tee -a "$LOG_FILE"
echo "Prompts:    ${PROMPTS[*]}"                       | tee -a "$LOG_FILE"
echo "Pretrains:  ${PRETRAINS[*]}"                     | tee -a "$LOG_FILE"
echo "Epochs:     $EPOCHS  (fast=$FAST)"               | tee -a "$LOG_FILE"
echo "Device:     $DEVICE"                             | tee -a "$LOG_FILE"
echo "Num iter:   ${NUM_ITER:-bench-default}"          | tee -a "$LOG_FILE"
echo "Allow miss: $ALLOW_MISSING"                      | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                           | tee -a "$LOG_FILE"
echo                                                   | tee -a "$LOG_FILE"

echo "--- Bootstrap Excel templates ---" | tee -a "$LOG_FILE"
python create_excel_for_bench.py 2>&1 | tee -a "$LOG_FILE" || true
echo | tee -a "$LOG_FILE"

PASS=()
FAIL=()
SKIP=()

# Build dataset / task pairs --------------------------------------------------
declare -a CASES   # entries: "<pretrain_task>|<dataset>"
add_cases() {
    local task_kind="$1"; shift
    local ds_list=("$@")
    if [[ "${#DATASETS[@]}" -gt 0 ]]; then
        for d in "${ds_list[@]}"; do
            for filter in "${DATASETS[@]}"; do
                if [[ "$d" == "$filter" ]]; then
                    CASES+=("${task_kind}|${d}")
                fi
            done
        done
    else
        for d in "${ds_list[@]}"; do
            CASES+=("${task_kind}|${d}")
        done
    fi
}
case "$TASK" in
    node)  add_cases NodeTask  "${PAPER_NODE_DATASETS[@]}"  ;;
    graph) add_cases GraphTask "${PAPER_GRAPH_DATASETS[@]}" ;;
    all)
        add_cases NodeTask  "${PAPER_NODE_DATASETS[@]}"
        add_cases GraphTask "${PAPER_GRAPH_DATASETS[@]}"
        ;;
    *) echo "--task must be node|graph|all" >&2; exit 1 ;;
esac

for shot in "${SHOTS[@]}"; do
    for entry in "${CASES[@]}"; do
        IFS='|' read -r task_kind dataset <<< "$entry"
        for prompt in "${PROMPTS[@]}"; do
            for pretrain in "${PRETRAINS[@]}"; do
                # Paper convention: skip (pretrain != None) with (prompt == None)
                # — that cell is not in Table 2/3.
                if [[ "$prompt" == "None" && "$pretrain" != "None" ]]; then
                    continue
                fi

                label="${task_kind}/${dataset}/${shot}shot/${pretrain}+${prompt}"
                ckpt="$REPO_ROOT/Experiment/pre_trained_model/${dataset}/${pretrain}.GCN.128hidden_dim.pth"
                ckpt_arg="None"

                if [[ "$pretrain" != "None" ]]; then
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
                    --gnn_type      GCN
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
