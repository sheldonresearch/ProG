"""Smoke tests for the LinkTask pipeline (Phase 5 — LinkTask v1).

Coverage:
- ``LinkTask`` is reachable from ``prompt_graph.tasker`` and dispatched by
  ``bench.do_config_bench``.
- Single-graph LP (Cora) and multi-graph LP (MUTAG) each complete one run
  and return a valid 9-tuple with bounded metrics.
- Few-shot path (``shot_num > 0``) produces the same shape.
- Tier-3 prompts (GPPT, MultiGprompt, Prodigy, RELIEF) raise
  ``NotImplementedError`` at construction so callers fail fast — this is a
  capability check, not an ``xfail``: the strategies have NO LinkTask
  adapter yet, and would silently produce invalid metrics under a default
  encoder fallback.
- The Tier 1+2 strategies that DO have LinkTask encoders (12 in total) each
  complete a 1-epoch smoke run on Cora.
"""

from __future__ import annotations

import argparse

import pytest

bench = pytest.importorskip("bench")
downstream_task = pytest.importorskip("downstream_task")

from prompt_graph.tasker import LinkTask  # noqa: E402
from prompt_graph.tasker.link_task import (  # noqa: E402
    LINK_OUTPUT_DIM,
    LINK_TASK_SUPPORTED_PROMPTS,
    LINK_TASK_UNSUPPORTED_PROMPTS,
)


def _ns(**overrides):
    base = dict(
        pretrain_task="LinkTask",
        dataset_name="Cora",
        prompt_type="None",
        gnn_type="GCN",
        num_layer=2,
        hid_dim=128,
        epochs=1,
        shot_num=0,
        device="cpu",
        pre_train_model_path="None",
        batch_size=64,
        lr=0.001,
        decay=0.0,
        seed=42,
        num_iter=1,
        task_num=1,
        aio_num_hops=1,
        aio_max_nodes=20,
        aio_max_train_edges=12,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _assert_valid_link_result(result):
    assert result is not None
    assert result.pretrain_task_type == "LinkTask"
    for field in (
        result.final_acc_mean,
        result.final_f1_mean,
        result.final_roc_mean,
        result.final_prc_mean,
    ):
        assert 0.0 <= float(field) <= 1.0


# ---------------------------------------------------------------------------
# Registry + dispatch smoke
# ---------------------------------------------------------------------------


def test_link_task_exported_from_tasker():
    """LinkTask must be importable from the public tasker package."""
    from prompt_graph.tasker import LinkTask as Imported  # noqa: F401

    assert Imported is LinkTask


def test_link_task_supported_unsupported_disjoint():
    """The two sets MUST partition the strategies LinkTask knows about."""
    assert not (LINK_TASK_SUPPORTED_PROMPTS & LINK_TASK_UNSUPPORTED_PROMPTS)


def test_link_task_downstream_delegate():
    """downstream_task.py must support the public LinkTask entry point."""
    args = _ns(downstream_task="LinkTask", prompt_type="None", epochs=1)
    tasker = downstream_task.get_downstream_task_delegate(args)

    assert isinstance(tasker, LinkTask)
    assert tasker.output_dim == LINK_OUTPUT_DIM


def test_link_task_unsupported_prompts_raise():
    """Tier-3 strategies without LinkTask adapters raise at construction."""
    for p in sorted(LINK_TASK_UNSUPPORTED_PROMPTS):
        args = _ns(prompt_type=p)
        with pytest.raises(NotImplementedError) as exc_info:
            bench.do_config_bench(args)
        assert "LinkTask does not yet support" in str(exc_info.value)


# ---------------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------------


def test_link_task_cora_none_full_split():
    """Full-split (shot_num=0) baseline on Cora."""
    args = _ns(prompt_type="None", shot_num=0, epochs=2)
    result = bench.do_config_bench(args)
    _assert_valid_link_result(result)
    # With 2 epochs from a fresh GNN, untrained AUROC ought to clear ~0.55
    # comfortably; we assert > 0.55 as a sanity gate.
    assert result.final_roc_mean > 0.55, (
        f"Cora LinkTask None should clear AUROC > 0.55 but got {result.final_roc_mean:.4f}"
    )


def test_link_task_cora_none_few_shot():
    """Few-shot (shot_num=5) Cora — subsampled positives + RandomLinkSplit."""
    args = _ns(prompt_type="None", shot_num=5, epochs=2)
    result = bench.do_config_bench(args)
    _assert_valid_link_result(result)


def test_link_task_mutag_multi_graph():
    """Multi-graph LP on MUTAG using per-graph negative sampling."""
    args = _ns(dataset_name="MUTAG", prompt_type="None", shot_num=0, epochs=2)
    result = bench.do_config_bench(args)
    _assert_valid_link_result(result)


# Parametrized strategy smoke -------------------------------------------------


@pytest.mark.parametrize("prompt_type", sorted(LINK_TASK_SUPPORTED_PROMPTS))
def test_link_task_supported_strategy_smoke(prompt_type):
    """Every supported strategy, including All-in-one adapter, completes smoke LP."""
    args = _ns(prompt_type=prompt_type, epochs=1)
    result = bench.do_config_bench(args)
    _assert_valid_link_result(result)
