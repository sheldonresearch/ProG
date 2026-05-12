"""Strategy framework for prompt-type-specific training/evaluation logic.

Phase 4 foundation: this module introduces the ``PromptStrategy`` protocol,
a shared ``TaskContext`` payload, and a process-wide registry. Concrete
strategy classes are added in subsequent Phase 4 units and registered via
``@register_strategy('<name>')`` on import (see
``prompt_graph/tasker/strategies/__init__.py``).

See ``Docs/IMPROVEMENTS.md`` section 6.4 for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Protocol, Tuple

import torch


@dataclass
class TaskContext:
    """Mutable bundle of state shared between ``BaseTask`` and a strategy.

    All fields default to ``None`` so a strategy can be unit-tested without
    constructing a full task. ``BaseTask`` populates the relevant fields
    before delegating to ``setup`` / ``train_epoch`` / ``evaluate``.
    """

    gnn: Any = field(default=None)
    prompt: Any = field(default=None)
    answering: Any = field(default=None)
    device: Any = field(default=None)
    hid_dim: int = field(default=None)
    output_dim: int = field(default=None)
    criterion: Any = field(default=None)
    optimizer: Any = field(default=None)
    pg_opi: Any = field(default=None)
    answer_opi: Any = field(default=None)
    data: Any = field(default=None)
    dataset_name: str = field(default=None)


class PromptStrategy(Protocol):
    """Protocol every prompt-type strategy must satisfy.

    Implementations are plain classes (not ABCs) registered through
    ``register_strategy``. ``name`` is the registry key and must match the
    ``prompt_type`` CLI flag.
    """

    name: ClassVar[str]

    def setup(self, ctx: TaskContext) -> None:
        """Construct the prompt module and attach it to ``ctx``."""
        ...

    def configure_optimizer(self, ctx: TaskContext) -> torch.optim.Optimizer:
        """Build the optimizer(s) and return the primary one."""
        ...

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        """Run one training epoch; return the mean loss."""
        ...

    def evaluate(self, ctx: TaskContext, loader_or_data) -> Tuple[float, float, float, float]:
        """Evaluate on the given loader/data; return ``(acc, f1, roc, prc)``."""
        ...


STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(name: str):
    """Class decorator that registers a strategy under ``name``.

    Raises ``ValueError`` if ``name`` is already taken so duplicate
    registrations are surfaced at import time instead of silently shadowing.
    """

    def decorator(cls: type) -> type:
        if name in STRATEGY_REGISTRY:
            raise ValueError(
                f"Strategy '{name}' is already registered to "
                f"{STRATEGY_REGISTRY[name].__name__}"
            )
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str):
    """Return the strategy class registered under ``name``.

    Raises ``KeyError`` with a message listing the currently registered
    strategies so callers get an actionable error.
    """
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        available = sorted(STRATEGY_REGISTRY) or ['<none registered>']
        raise KeyError(
            f"No strategy registered under '{name}'. "
            f"Available strategies: {available}"
        )
