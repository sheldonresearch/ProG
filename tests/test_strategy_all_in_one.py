"""Smoke tests for the All-in-one Strategy migration (Phase 4 Unit 18)."""

import argparse

import pytest

bench = pytest.importorskip("bench")


def _ns(**overrides):
    base = dict(
        pretrain_task="NodeTask",
        dataset_name="Cora",
        prompt_type="All-in-one",
        gnn_type="GCN",
        num_layer=2,
        hid_dim=128,
        epochs=1,
        shot_num=5,
        device="cpu",
        pre_train_model_path="None",
        batch_size=64,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_strategy_registry_has_all_in_one():
    """All-in-one must be registered after import."""
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert "All-in-one" in STRATEGY_REGISTRY


def test_node_task_cora_all_in_one_strategy():
    """NodeTask + Cora + All-in-one must complete and route through Strategy."""
    args = _ns(prompt_type="All-in-one")
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0


def test_graph_task_mutag_all_in_one_strategy():
    """GraphTask + MUTAG + All-in-one must complete via Strategy."""
    args = _ns(
        pretrain_task="GraphTask",
        dataset_name="MUTAG",
        prompt_type="All-in-one",
    )
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
