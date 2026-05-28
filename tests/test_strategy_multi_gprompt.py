"""Smoke tests for the MultiGprompt Strategy migration (Phase 4 Unit 20).

MultiGprompt is the only prompt_type that cannot be exercised end-to-end
without a real pretrain checkpoint on disk (``task.py`` always calls
``torch.load(self.pre_train_model_path)`` for this branch and there is no
``pre_train_model_path='None'`` short-circuit). The bench-level smoke
test is therefore skipped if no checkpoint is wired up — what we verify
here is that the strategy is registered, instantiable, and exposes the
``train_epoch`` / ``evaluate`` methods the dispatcher relies on.
"""

import argparse
import os

import pytest

bench = pytest.importorskip("bench")


def _ns(**overrides):
    base = dict(
        pretrain_task="NodeTask",
        dataset_name="Cora",
        prompt_type="MultiGprompt",
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


def test_strategy_registry_has_multigprompt():
    """MultiGprompt must be registered after import."""
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert "MultiGprompt" in STRATEGY_REGISTRY


def test_multigprompt_strategy_interface():
    """The registered class must implement train_epoch and evaluate."""
    from prompt_graph.tasker.strategy import get_strategy

    cls = get_strategy("MultiGprompt")
    inst = cls()
    assert callable(getattr(inst, "train_epoch", None))
    assert callable(getattr(inst, "evaluate", None))


def test_node_task_cora_multigprompt_strategy():
    """NodeTask + Cora + MultiGprompt must complete and route through Strategy.

    Requires a real pretrain checkpoint via the
    ``MULTIGPROMPT_PRETRAIN_PATH`` env var. Skipped in CI when unset.
    """
    pretrain_path = os.environ.get("MULTIGPROMPT_PRETRAIN_PATH")
    if not pretrain_path or not os.path.exists(pretrain_path):
        pytest.skip(
            "MultiGprompt requires a pretrain checkpoint; set "
            "MULTIGPROMPT_PRETRAIN_PATH to a valid .pth file to enable."
        )
    args = _ns(prompt_type="MultiGprompt", pre_train_model_path=pretrain_path)
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
