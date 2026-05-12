"""Smoke tests for the GPF / GPF-plus Strategy migration (Phase 4 Unit 16)."""
import argparse

import pytest


bench = pytest.importorskip('bench')


def _ns(**overrides):
    base = dict(
        pretrain_task='NodeTask',
        dataset_name='Cora',
        prompt_type='GPF',
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


def test_strategy_registry_has_gpf():
    """GPF and GPF-plus must be registered after import."""
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert 'GPF' in STRATEGY_REGISTRY
    assert 'GPF-plus' in STRATEGY_REGISTRY


def test_node_task_cora_gpf_strategy():
    """NodeTask + Cora + GPF must complete and route through Strategy."""
    args = _ns(prompt_type='GPF')
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0


def test_graph_task_mutag_gpf_strategy():
    """GraphTask + MUTAG + GPF must complete via Strategy."""
    args = _ns(
        pretrain_task='GraphTask',
        dataset_name='MUTAG',
        prompt_type='GPF',
    )
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
