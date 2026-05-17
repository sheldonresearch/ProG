"""GraphPrompterStrategy: prompt_type == 'GraphPrompter' train/eval logic.

GraphPrompter (KDD 2025) learns a scoring MLP over candidate prompt
supernodes.  This ProG port keeps the scoring MLP and prompt bank, dropping
the kNN cache and text-embedding components.
"""

from __future__ import annotations

import torch.nn.functional as F

from prompt_graph.evaluation import GNNGraphEva, GNNNodeEva

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("GraphPrompter")
class GraphPrompterStrategy(PromptStrategy):
    """Adaptive prompt selection via scoring MLP."""

    def setup(self, ctx: TaskContext) -> None:
        """No-op: prompt is built by BaseTask.initialize_prompt."""

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """No-op: optimizer is built by BaseTask.initialize_optimizer."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            return self._train_node(ctx, loader_or_data)
        return self._train_graph(ctx, loader_or_data)

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            data, idx_test = loader_or_data
            return GNNNodeEva(data, idx_test, ctx.gnn, ctx.answering, ctx.output_dim, ctx.device)
        return GNNGraphEva(loader_or_data, ctx.gnn, ctx.answering, ctx.output_dim, ctx.device)

    @staticmethod
    def _train_node(ctx: TaskContext, data_and_idx) -> float:
        data, train_idx = data_and_idx
        ctx.prompt.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()
        out = ctx.gnn(data.x, data.edge_index, batch=None, prompt=ctx.prompt, prompt_type="GraphPrompter")
        out = ctx.answering(out)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    @staticmethod
    def _train_graph(ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        ctx.answering.train()
        total_loss = 0.0
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            out = ctx.gnn(batch.x, batch.edge_index, batch.batch, prompt=ctx.prompt, prompt_type="GraphPrompter")
            out = ctx.answering(out)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
