#!/usr/bin/env bash
# Per-shard runner for Plan A on amlt Singularity workers (Feeds, 1x A100).
#
# Each amulet job sets these env vars and then execs this script:
#   SHARD_NAME           — label for tags / logs
#   PRETRAIN_METHODS     — space-sep methods to (re)train, e.g. "DGI GraphCL GraphMAE MultiGprompt"
#   PRETRAIN_DATASETS    — space-sep datasets, e.g. "Cora MUTAG"
#   BENCH_PROMPTS        — space-sep prompts for this shard, e.g. "GPF GPF-plus"
#   BENCH_DATASETS       — space-sep datasets for bench (subset of PRETRAIN_DATASETS)
#   BENCH_PRETRAINS      — space-sep pretrains (default: "None DGI GraphCL GraphMAE MultiGprompt")
#   BENCH_SHOTS          — space-sep shots (default: "1")
#   BENCH_EXTRA_FLAGS    — extra bench flags (e.g. "--include-slow --num-iter 1")
#
# Worker image: amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04
# (which actually ships torch 2.7.1+cu126 — see msr-amlt-feeds skill §4.12).
# /opt/conda/envs/ptca is read-only, so we install everything with --user.

set -euo pipefail

: "${SHARD_NAME:?SHARD_NAME must be set}"
: "${PRETRAIN_METHODS:?PRETRAIN_METHODS must be set}"
: "${PRETRAIN_DATASETS:?PRETRAIN_DATASETS must be set}"
: "${BENCH_PROMPTS:?BENCH_PROMPTS must be set}"
: "${BENCH_DATASETS:?BENCH_DATASETS must be set}"

BENCH_PRETRAINS="${BENCH_PRETRAINS:-None DGI GraphCL GraphMAE MultiGprompt}"
BENCH_SHOTS="${BENCH_SHOTS:-1}"
BENCH_EXTRA_FLAGS="${BENCH_EXTRA_FLAGS:-}"

echo "=== Shard: ${SHARD_NAME} ==="
echo "Date:               $(date -u +%FT%TZ)"
echo "AMLT_CODE_DIR:      ${AMLT_CODE_DIR:-<unset>}"
echo "AMLT_OUTPUT_DIR:    ${AMLT_OUTPUT_DIR:-<unset>}"
echo "PRETRAIN_METHODS:   ${PRETRAIN_METHODS}"
echo "PRETRAIN_DATASETS:  ${PRETRAIN_DATASETS}"
echo "BENCH_PROMPTS:      ${BENCH_PROMPTS}"
echo "BENCH_DATASETS:     ${BENCH_DATASETS}"
echo "BENCH_PRETRAINS:    ${BENCH_PRETRAINS}"
echo "BENCH_SHOTS:        ${BENCH_SHOTS}"
echo "BENCH_EXTRA_FLAGS:  ${BENCH_EXTRA_FLAGS}"
echo

cd "${AMLT_CODE_DIR}"

echo "--- 1. Install python deps to ~/.local (--user, RO ptca env) ---"
python --version
python -m pip install --user --upgrade pip >/dev/null
# Core pyg + analysis stack.
python -m pip install --user --no-warn-script-location \
    "torch_geometric>=2.5,<2.7" \
    "ogb" "pandas" "openpyxl" "scikit-learn" \
    "torchmetrics" "networkx" "numpy<2" "deprecated" \
    "tqdm" "pyyaml" "outdated"
# torch_scatter is a hard top-level import of RELIEF + GraphPrompter
# strategies (loaded eagerly by tasker/strategies/__init__.py). torch_sparse
# is required by torch_geometric.loader.ClusterData (used by NodePretrain's
# METIS-partition path → triggered by GraphCL/GraphMAE on node datasets
# like Cora). The worker image ships torch 2.7.1+cu126; use the
# pyg-published prebuilt wheels for torch 2.7.x + cu126 to avoid a
# multi-minute source build (and to dodge the missing CUDA-headers issue
# in the ptca image).
python -m pip install --user --no-warn-script-location \
    --no-build-isolation --no-cache-dir \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html \
    "torch_scatter==2.1.2" "torch_sparse==0.6.18"

echo
echo "--- 2. Sanity-check imports ---"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"
# Expose the local repo (no need for `pip install -e .` since /opt is RO).
export PYTHONPATH="${AMLT_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "from prompt_graph.utils import resolve_device; from prompt_graph.tasker.strategy import STRATEGY_REGISTRY; import prompt_graph.tasker.strategies as _s; print('prompt_graph OK; strategies:', sorted(STRATEGY_REGISTRY))"

echo
echo "--- 3. Pretrain (idempotent skip; tolerate per-pretrain failures) ---"
# pretrain_full_grid.sh exits 1 if any single pretrain failed; we don't want
# that to kill the whole shard — bench can still proceed with whatever
# checkpoints did get created. set +e to capture the rc; report it but
# continue. Truly catastrophic failures (no checkpoints at all) will be
# caught by bench_full_grid.sh complaining about missing .pth files.
set +e
bash scripts/pretrain_full_grid.sh \
    --device 0 \
    --fast \
    --methods "${PRETRAIN_METHODS}" \
    --datasets "${PRETRAIN_DATASETS}" \
    --tag "${SHARD_NAME}-pre"
PRETRAIN_RC=$?
set -e
if [[ "${PRETRAIN_RC}" -ne 0 ]]; then
    echo "WARN: pretrain_full_grid.sh exited rc=${PRETRAIN_RC} (some pretrain methods failed); continuing to bench" >&2
fi

echo
echo "--- 4. Bench (tolerate per-cell failures) ---"
# Same tolerance pattern as pretrain — bench_full_grid.sh exits 1 if any
# single (prompt × pretrain × shot) cell fails, but we still want the
# excel rows for the cells that did succeed.
# shellcheck disable=SC2086  # intentional word-splitting on BENCH_EXTRA_FLAGS
set +e
bash scripts/bench_full_grid.sh \
    --device 0 \
    --fast \
    --prompts "${BENCH_PROMPTS}" \
    --datasets "${BENCH_DATASETS}" \
    --pretrains "${BENCH_PRETRAINS}" \
    --shots "${BENCH_SHOTS}" \
    --tag "${SHARD_NAME}" \
    ${BENCH_EXTRA_FLAGS}
BENCH_RC=$?
set -e
if [[ "${BENCH_RC}" -ne 0 ]]; then
    echo "WARN: bench_full_grid.sh exited rc=${BENCH_RC} (some bench cells failed); preserving partial outputs anyway" >&2
fi

echo
echo "--- 5. Preserve outputs to AMLT_OUTPUT_DIR ---"
OUT="${AMLT_OUTPUT_DIR:-/tmp/amlt_out_${SHARD_NAME}}"
mkdir -p "${OUT}/excel" "${OUT}/logs" "${OUT}/pretrained"
if [[ -d Experiment/ExcelResults ]]; then
    cp -r Experiment/ExcelResults/. "${OUT}/excel/"
    echo "Copied Excel results -> ${OUT}/excel/"
fi
if [[ -d scripts/baseline_logs ]]; then
    cp -r scripts/baseline_logs/. "${OUT}/logs/"
    echo "Copied logs -> ${OUT}/logs/"
fi
if [[ -d Experiment/pre_trained_model ]]; then
    cp -r Experiment/pre_trained_model/. "${OUT}/pretrained/"
    echo "Copied pretrained ckpts -> ${OUT}/pretrained/"
fi

# Write a tiny shard manifest so post-merge can identify what was produced.
{
    echo "shard_name: ${SHARD_NAME}"
    echo "bench_prompts: ${BENCH_PROMPTS}"
    echo "bench_datasets: ${BENCH_DATASETS}"
    echo "bench_pretrains: ${BENCH_PRETRAINS}"
    echo "bench_shots: ${BENCH_SHOTS}"
    echo "bench_extra_flags: ${BENCH_EXTRA_FLAGS}"
    echo "completed_at: $(date -u +%FT%TZ)"
} > "${OUT}/SHARD_MANIFEST.yaml"

echo "=== Shard ${SHARD_NAME} DONE ==="
