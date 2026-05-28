#!/usr/bin/env bash
# Bulk-pretrain script for reproducing the "Overall Performance" experiment
# (Section 5.1 of the ProG paper, arXiv:2406.05346).
#
# Iterates 6 pretrain methods × 15 paper datasets = up to 90 checkpoints.
# Idempotent: skips combos whose .pth already exists under
#   Experiment/pre_trained_model/<dataset>/<method>.<gnn_type>.128hidden_dim.pth
#
# This is the prerequisite for scripts/bench_overall_performance.sh — bench.py
# expects those .pth files to exist for every (pretrain_method, dataset) cell.
#
# Usage:
#   bash scripts/pretrain_paper_grid.sh                          # full run
#   bash scripts/pretrain_paper_grid.sh --fast                   # 200 epochs
#   bash scripts/pretrain_paper_grid.sh --task node              # node-only
#   bash scripts/pretrain_paper_grid.sh --task graph             # graph-only
#   bash scripts/pretrain_paper_grid.sh --methods "DGI GraphCL"  # subset
#   bash scripts/pretrain_paper_grid.sh --datasets "Cora MUTAG"  # subset
#   bash scripts/pretrain_paper_grid.sh --device 0               # GPU 0
#   bash scripts/pretrain_paper_grid.sh --gnn_type GAT           # backbone
#   bash scripts/pretrain_paper_grid.sh --epochs 500             # override
#
# Output:
#   - Checkpoints: Experiment/pre_trained_model/<dataset>/<method>.<gnn_type>.128hidden_dim.pth
#   - Log: scripts/baseline_logs/${TAG}_${STAMP}_pretrain_grid.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- paper grid -------------------------------------------------------------
PAPER_NODE_DATASETS=(Cora CiteSeer PubMed Wisconsin Texas Actor ogbn-arxiv)
PAPER_GRAPH_DATASETS=(MUTAG IMDB-BINARY COLLAB PROTEINS ENZYMES COX2 BZR DD)
PAPER_PRETRAINS=(DGI GraphMAE Edgepred_GPPT Edgepred_Gprompt GraphCL SimGRACE)

# ---- defaults ---------------------------------------------------------------
TASK="all"
EPOCHS=1000
DEVICE="cpu"
GNN_TYPE="GCN"
TAG="pretrain-grid"
METHODS=()
DATASETS=()
FAST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     TASK="$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --gnn_type|--gnn-type) GNN_TYPE="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --fast)     FAST=1; shift ;;
        --methods)  read -ra METHODS  <<< "$2"; shift 2 ;;
        --datasets) read -ra DATASETS <<< "$2"; shift 2 ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS=200
fi

if [[ "${#METHODS[@]}" -eq 0 ]]; then
    METHODS=("${PAPER_PRETRAINS[@]}")
fi
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    case "$TASK" in
        node)  DATASETS=("${PAPER_NODE_DATASETS[@]}") ;;
        graph) DATASETS=("${PAPER_GRAPH_DATASETS[@]}") ;;
        all)   DATASETS=("${PAPER_NODE_DATASETS[@]}" "${PAPER_GRAPH_DATASETS[@]}") ;;
        *) echo "--task must be node|graph|all, got $TASK" >&2; exit 1 ;;
    esac
fi

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}_pretrain_grid.log"

echo "=== Pretrain grid sweep ==="           | tee "$LOG_FILE"
echo "Tag:      $TAG"                         | tee -a "$LOG_FILE"
echo "Task:     $TASK"                        | tee -a "$LOG_FILE"
echo "Epochs:   $EPOCHS  (fast=$FAST)"        | tee -a "$LOG_FILE"
echo "Device:   $DEVICE"                      | tee -a "$LOG_FILE"
echo "GNN type: $GNN_TYPE"                    | tee -a "$LOG_FILE"
echo "Methods:  ${METHODS[*]}"                | tee -a "$LOG_FILE"
echo "Datasets: ${DATASETS[*]}"               | tee -a "$LOG_FILE"
echo "Log:      $LOG_FILE"                    | tee -a "$LOG_FILE"
echo                                          | tee -a "$LOG_FILE"

PASS=()
FAIL=()
SKIP=()

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        ckpt="$REPO_ROOT/Experiment/pre_trained_model/${dataset}/${method}.${GNN_TYPE}.128hidden_dim.pth"
        label="${method} @ ${dataset}"
        if [[ -f "$ckpt" ]]; then
            echo "--- SKIP (exists): $label -> $ckpt ---" | tee -a "$LOG_FILE"
            SKIP+=("$label")
            continue
        fi
        echo "--- $label ---" | tee -a "$LOG_FILE"
        cmd=(
            python pre_train.py
            --pretrain_task "$method"
            --dataset_name  "$dataset"
            --gnn_type      "$GNN_TYPE"
            --hid_dim       128
            --num_layer     2
            --epochs        "$EPOCHS"
            --device        "$DEVICE"
            --seed          42
        )
        echo "+ ${cmd[*]}" | tee -a "$LOG_FILE"
        if "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            PASS+=("$label")
        else
            FAIL+=("$label")
        fi
        echo | tee -a "$LOG_FILE"
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
