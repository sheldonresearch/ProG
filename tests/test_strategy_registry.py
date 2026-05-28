"""Unit tests for the Phase 4 strategy registry foundation."""

import pytest

from prompt_graph.tasker.strategy import (
    STRATEGY_REGISTRY,
    TaskContext,
    get_strategy,
    register_strategy,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore the global registry around each test."""
    saved = STRATEGY_REGISTRY.copy()
    try:
        yield
    finally:
        STRATEGY_REGISTRY.clear()
        STRATEGY_REGISTRY.update(saved)


def test_register_strategy_adds_to_registry():
    @register_strategy("X")
    class DummyStrategy:
        pass

    assert STRATEGY_REGISTRY["X"] is DummyStrategy
    assert DummyStrategy.name == "X"


def test_get_strategy_returns_registered_class():
    @register_strategy("Y")
    class AnotherStrategy:
        pass

    assert get_strategy("Y") is AnotherStrategy


def test_get_strategy_missing_raises_with_available_names():
    @register_strategy("Z")
    class ZStrategy:
        pass

    with pytest.raises(KeyError) as excinfo:
        get_strategy("NonExistent")

    message = str(excinfo.value)
    assert "NonExistent" in message
    assert "Z" in message


def test_register_strategy_rejects_duplicate_names():
    @register_strategy("Dup")
    class FirstStrategy:
        pass

    with pytest.raises(ValueError):

        @register_strategy("Dup")
        class SecondStrategy:
            pass


def test_task_context_can_be_instantiated_empty():
    ctx = TaskContext()
    assert ctx.gnn is None
    assert ctx.prompt is None
    assert ctx.dataset_name is None


def test_task_context_accepts_fields():
    ctx = TaskContext(hid_dim=128, output_dim=7, dataset_name="Cora")
    assert ctx.hid_dim == 128
    assert ctx.output_dim == 7
    assert ctx.dataset_name == "Cora"


def test_all_bundled_strategies_register_on_import():
    """Importing ``prompt_graph.tasker.strategies`` must register every
    bundled strategy. This catches the common mistake of adding a strategy
    file but forgetting the ``from . import <name>`` line in
    ``strategies/__init__.py``.

    If this list legitimately changes (added or removed a strategy), update
    ``EXPECTED`` below and the public README's supported strategy list in the same PR.
    """
    # Re-import to fire the registration side effect even if a prior test
    # cleared the registry.
    import importlib

    from prompt_graph.tasker import strategies as strategies_pkg

    importlib.reload(strategies_pkg)

    EXPECTED = {
        "None",
        "GPF",
        "GPF-plus",
        "Gprompt",
        "All-in-one",
        "GPPT",
        "MultiGprompt",
        "Prodigy",
        "UniPrompt",
        "SelfPro",
        "ProNoG",
        "DAGPrompT",
        "PSP",
        "RELIEF",
        "GraphPrompter",
        "EdgePrompt",
        "EdgePromptplus",
    }
    missing = EXPECTED - set(STRATEGY_REGISTRY.keys())
    assert not missing, (
        f"Strategies missing from STRATEGY_REGISTRY: {sorted(missing)}. "
        f"Check strategies/__init__.py imports."
    )
