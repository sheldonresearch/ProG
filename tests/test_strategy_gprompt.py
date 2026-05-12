"""Smoke tests for the Gprompt Strategy migration (Phase 4 Unit 17)."""
import argparse

import pytest


bench = pytest.importorskip('bench')


def _ns(**overrides):
    base = dict(
        pretrain_task='NodeTask',
        dataset_name='Cora',
        prompt_type='Gprompt',
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


def test_strategy_registry_has_gprompt():
    """Gprompt must be registered after import."""
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert 'Gprompt' in STRATEGY_REGISTRY


def test_strategy_fresh_instance_has_none_centers():
    """A freshly constructed strategy must not carry state from another instance."""
    from prompt_graph.tasker.strategy import get_strategy

    strategy = get_strategy('Gprompt')()
    assert strategy.mean_centers is None


def test_node_task_cora_gprompt_strategy():
    """NodeTask + Cora + Gprompt must complete and route through Strategy."""
    args = _ns(prompt_type='Gprompt')
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0


def test_graph_task_mutag_gprompt_strategy():
    """GraphTask + MUTAG + Gprompt must complete via Strategy."""
    args = _ns(
        pretrain_task='GraphTask',
        dataset_name='MUTAG',
        prompt_type='Gprompt',
    )
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
