"""End-to-end smoke test for the bench pipeline."""
import argparse

import pytest

# Best-effort import — if bench module not importable, skip the test entirely.
bench = pytest.importorskip('bench')


def test_do_config_bench_node_cora_gpf():
    """1-epoch Cora + GPF NodeTask must complete and return valid metrics."""
    args = argparse.Namespace(
        pretrain_task='NodeTask',
        dataset_name='Cora',
        prompt_type='GPF',
        gnn_type='GCN',
        num_layer=2,
        hid_dim=128,
        epochs=1,
        shot_num=5,
        device=0,
        pre_train_model_path='None',
        batch_size=64,
        seed=42,
    )
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
    assert 0.0 <= result.final_f1_mean <= 1.0
