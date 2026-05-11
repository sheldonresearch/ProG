#!/usr/bin/env bash
# Phase 0 金标准命令：固定 3 组覆盖三种数据规模的 bench 调用。
# 每个 Phase 合并前都跑一次，把指标更新到 Docs/baseline_metrics.md。
#
# 用法：
#   bash scripts/baseline.sh                # 跑完整 baseline（耗时较长）
#   bash scripts/baseline.sh --fast         # 快速回归（缩短 epochs）
#   bash scripts/baseline.sh --tag phase-1  # 指定写入 metric 表格的列名
#
# 输出：stdout 同时落盘到 scripts/baseline_logs/<tag>_<datetime>.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FAST=0
TAG="manual"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST=1
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$FAST" -eq 1 ]]; then
    EPOCHS_NODE=50
    EPOCHS_GRAPH=50
else
    EPOCHS_NODE=200
    EPOCHS_GRAPH=200
fi

LOG_DIR="$REPO_ROOT/scripts/baseline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${TAG}_${STAMP}.log"

echo "=== Phase 0 baseline run ===" | tee "$LOG_FILE"
echo "Tag:       $TAG"             | tee -a "$LOG_FILE"
echo "Fast mode: $FAST"             | tee -a "$LOG_FILE"
echo "Repo:      $REPO_ROOT"        | tee -a "$LOG_FILE"
echo "Log:       $LOG_FILE"         | tee -a "$LOG_FILE"
echo                                | tee -a "$LOG_FILE"

# Excel templates are required by bench.py; bootstrap once.
echo "--- Bootstrap Excel templates ---" | tee -a "$LOG_FILE"
python create_excel_for_bench.py 2>&1 | tee -a "$LOG_FILE"
echo                                    | tee -a "$LOG_FILE"

run_case() {
    local label="$1"; shift
    echo "--- $label ---" | tee -a "$LOG_FILE"
    echo "+ python bench.py $*" | tee -a "$LOG_FILE"
    /usr/bin/env time -p python bench.py "$@" 2>&1 | tee -a "$LOG_FILE"
    echo | tee -a "$LOG_FILE"
}

# Case 1 — Cora / GPF (NodeTask, smallest dataset)
run_case "Case 1: Cora + GraphCL + GPF (NodeTask)" \
    --pretrain_task NodeTask \
    --dataset_name  Cora \
    --prompt_type   GPF \
    --gnn_type      GCN \
    --shot_num      5 \
    --seed          42 \
    --epochs        "$EPOCHS_NODE" \
    --pre_train_model_path ./Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth \
    "${EXTRA_ARGS[@]}"

# Case 2 — MUTAG / All-in-one (GraphTask, small molecular dataset)
run_case "Case 2: MUTAG + GraphCL + All-in-one (GraphTask)" \
    --pretrain_task GraphTask \
    --dataset_name  MUTAG \
    --prompt_type   All-in-one \
    --gnn_type      GCN \
    --shot_num      5 \
    --seed          42 \
    --epochs        "$EPOCHS_GRAPH" \
    --pre_train_model_path ./Experiment/pre_trained_model/MUTAG/GraphCL.GCN.128hidden_dim.pth \
    "${EXTRA_ARGS[@]}"

# Case 3 — PubMed / Gprompt (NodeTask, mid-size citation network)
run_case "Case 3: PubMed + GraphCL + Gprompt (NodeTask)" \
    --pretrain_task NodeTask \
    --dataset_name  PubMed \
    --prompt_type   Gprompt \
    --gnn_type      GCN \
    --shot_num      1 \
    --seed          42 \
    --epochs        "$EPOCHS_NODE" \
    --pre_train_model_path ./Experiment/pre_trained_model/PubMed/GraphCL.GCN.128hidden_dim.pth \
    "${EXTRA_ARGS[@]}"

echo "=== baseline done. metrics excel: ./Experiment/ExcelResults/ ===" | tee -a "$LOG_FILE"
echo "Update Docs/baseline_metrics.md with the new column ($TAG)."      | tee -a "$LOG_FILE"
