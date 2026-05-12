"""Smoke tests for the GPPT Strategy migration (Phase 4 Unit 19)."""
import argparse

import pytest


bench = pytest.importorskip('bench')


def _ns(**overrides):
    base = dict(
        pretrain_task='NodeTask',
        dataset_name='Cora',
        prompt_type='GPPT',
        gnn_type='GCN',
        num_layer=2,
        hid_dim=128,
        epochs=1,
        shot_num=5,
        device='cpu',
        pre_train_model_path='None',
        batch_size=64,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_strategy_registry_has_gppt():
    """GPPT must be registered after import."""
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert 'GPPT' in STRATEGY_REGISTRY


def test_node_task_cora_gppt_strategy():
    """NodeTask + Cora + GPPT must complete and route through Strategy."""
    args = _ns(prompt_type='GPPT')
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0


def test_graph_task_mutag_gppt_strategy():
    """GraphTask + MUTAG + GPPT must complete via Strategy."""
    args = _ns(
        pretrain_task='GraphTask',
        dataset_name='MUTAG',
        prompt_type='GPPT',
    )
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
