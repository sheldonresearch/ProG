#!/usr/bin/env bash
# Bulk-pretrain ALL 7 pretrain methods × ALL 23 ProG datasets.
#
# Methods (7):
#   DGI, GraphMAE, Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE,
#   MultiGprompt
#
# Datasets (23 = NODE_TASKS (12) + GRAPH_TASKS (11)):
#   Node : Cora CiteSeer PubMed Wisconsin Texas Actor
#          Computers Photo Reddit WikiCS Flickr ogbn-arxiv
#   Graph: MUTAG ENZYMES COLLAB PROTEINS IMDB-BINARY REDDIT-BINARY
#          COX2 BZR PTC_MR DD ogbg-ppa
#
# Output file naming (matches bench.py expectations):
#   - Standard 6 methods:  Experiment/pre_trained_model/<dataset>/<method>.GCN.128hidden_dim.pth
#   - MultiGprompt:        Experiment/pre_trained_model/<dataset>/MultiGprompt.pth
#
# Idempotent: skips combos whose .pth already exists.
#
# Usage:
#   bash scripts/pretrain_full_grid.sh                                # full sweep, CPU
#   bash scripts/pretrain_full_grid.sh --device 0                     # GPU 0
#   bash scripts/pretrain_full_grid.sh --task node                    # node datasets only
#   bash scripts/pretrain_full_grid.sh --task graph                   # graph datasets only
#   bash scripts/pretrain_full_grid.sh --exclude-ogb                  # skip ogbn-arxiv + ogbg-ppa
#   bash scripts/pretrain_full_grid.sh --methods "DGI GraphCL"        # subset
#   bash scripts/pretrain_full_grid.sh --datasets "Cora MUTAG"        # subset
#   bash scripts/pretrain_full_grid.sh --fast                         # 200 epochs (default 1000)
#   bash scripts/pretrain_full_grid.sh --epochs 500                   # custom
#
# Output log:
#   scripts/baseline_logs/${TAG}_${STAMP}_pretrain_full.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- project grid (everything in defines.py) --------------------------------
ALL_NODE_DATASETS=(Cora CiteSeer PubMed Wisconsin Texas Actor Computers Photo Reddit WikiCS Flickr ogbn-arxiv)
ALL_GRAPH_DATASETS=(MUTAG ENZYMES COLLAB PROTEINS IMDB-BINARY REDDIT-BINARY COX2 BZR PTC_MR DD ogbg-ppa)
ALL_PRETRAINS=(DGI GraphMAE Edgepred_GPPT Edgepred_Gprompt GraphCL SimGRACE MultiGprompt)
OGB_DATASETS=(ogbn-arxiv ogbg-ppa)

# ---- defaults ---------------------------------------------------------------
TASK="all"
EPOCHS=1000
DEVICE="cpu"
TAG="pretrain-full"
EXCLUDE_OGB=0
FAST=0
METHODS=()
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)        TASK="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --fast)        FAST=1; shift ;;
        --exclude-ogb) EXCLUDE_OGB=1; shift ;;
        --methods)     read -ra METHODS  <<< "$2"; shift 2 ;;
        --datasets)    read -ra DATASETS <<< "$2"; shift 2 ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS=200
fi

if [[ "${#METHODS[@]}" -eq 0 ]]; then
    METHODS=("${ALL_PRETRAINS[@]}")
fi
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    case "$TASK" in
        node)  DATASETS=("${ALL_NODE_DATASETS[@]}") ;;
        graph) DATASETS=("${ALL_GRAPH_DATASETS[@]}") ;;
        all)   DATASETS=("${ALL_NODE_DATASETS[@]}" "${ALL_GRAPH_DATASETS[@]}") ;;
        *) echo "--task must be node|graph|all" >&2; exit 1 ;;
    esac
fi

is_ogb() {
    local d="$1"
    for ogb in "${OGB_DATASETS[@]}"; do
        [[ "$d" == "$ogb" ]] && return 0
    done
    return 1
}

ckpt_path() {
    local method="$1" dataset="$2"
    if [[ "$method" == "MultiGprompt" ]]; then
        echo "$REPO_ROOT/Experiment/pre_trained_model/${dataset}/MultiGprompt.pth"
    else
        echo "$REPO_ROOT/Experiment/pre_trained_model/${dataset}/${method}.GCN.128hidden_dim.pth"
    fi
}

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}_pretrain_full.log"

echo "=== Full pretrain grid sweep ==="            | tee "$LOG_FILE"
echo "Tag:         $TAG"                           | tee -a "$LOG_FILE"
echo "Task:        $TASK"                          | tee -a "$LOG_FILE"
echo "Epochs:      $EPOCHS  (fast=$FAST)"          | tee -a "$LOG_FILE"
echo "Device:      $DEVICE"                        | tee -a "$LOG_FILE"
echo "Methods:     ${METHODS[*]}"                  | tee -a "$LOG_FILE"
echo "Datasets:    ${DATASETS[*]}"                 | tee -a "$LOG_FILE"
echo "Exclude OGB: $EXCLUDE_OGB"                   | tee -a "$LOG_FILE"
echo "Log:         $LOG_FILE"                      | tee -a "$LOG_FILE"
echo                                               | tee -a "$LOG_FILE"

PASS=()
FAIL=()
SKIP=()

for dataset in "${DATASETS[@]}"; do
    if [[ "$EXCLUDE_OGB" -eq 1 ]] && is_ogb "$dataset"; then
        skip_label="* @ ${dataset}"
        echo "--- SKIP (--exclude-ogb): $skip_label ---" | tee -a "$LOG_FILE"
        SKIP+=("$skip_label")
        continue
    fi
    for method in "${METHODS[@]}"; do
        ckpt="$(ckpt_path "$method" "$dataset")"
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
            --gnn_type      GCN
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
