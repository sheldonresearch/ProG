"""RELIEFStrategy: prompt_type == 'RELIEF' train/eval logic.

RELIEF (NeurIPS 2024) uses an RL agent to build node-level feature
perturbations.  This ProG port replaces the PPO loop with direct gradient
descent on a learnable perturbation matrix.
"""

from __future__ import annotations

import torch.nn.functional as F

from prompt_graph.evaluation import GNNNodeEva

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("RELIEF")
class RELIEFStrategy(PromptStrategy):
    """RL-guided feature perturbation (simplified differentiable version)."""

    def setup(self, ctx: TaskContext) -> None:
        """No-op: prompt is built by BaseTask.initialize_prompt."""

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """No-op: optimizer is built by BaseTask.initialize_optimizer."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        data, train_idx = loader_or_data
        ctx.prompt.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()
        x_prompted = ctx.prompt(data.x)
        out = ctx.gnn(x_prompted, data.edge_index, batch=None)
        out = ctx.answering(out)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        data, idx_test = loader_or_data
        return GNNNodeEva(data, idx_test, ctx.gnn, ctx.answering, ctx.output_dim, ctx.device)
