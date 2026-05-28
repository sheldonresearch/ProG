"""NoneStrategy: prompt_type == 'None' train/eval logic.

Encapsulates the behaviour previously inlined in
``NodeTask.train`` / ``NodeTask.run`` and ``GraphTask.Train`` / ``GraphTask.run``
when ``prompt_type == 'None'``. A single class serves both node- and graph-level
tasks; the task type is read from ``ctx.extra['task_type']``.
"""

from __future__ import annotations

from torch import nn, optim

from prompt_graph.evaluation import GNNGraphEva, GNNNodeEva

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("None")
class NoneStrategy(PromptStrategy):
    """No-prompt baseline: train ``gnn + answering`` head end-to-end."""

    def setup(self, ctx: TaskContext) -> None:
        """Construct the answering head on ``ctx.device`` when missing."""
        if ctx.answering is None:
            ctx.answering = nn.Sequential(
                nn.Linear(ctx.hid_dim, ctx.output_dim),
                nn.Softmax(dim=1),
            ).to(ctx.device)

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """Adam over gnn + answering params; mirrors ``BaseTask.initialize_optimizer``."""
        lr = ctx.extra.get("lr", 1e-3)
        wd = ctx.extra.get("wd", 5e-4)
        param_groups = [
            {"params": ctx.gnn.parameters()},
            {"params": ctx.answering.parameters()},
        ]
        ctx.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=wd)

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            return self._train_node(ctx, loader_or_data)
        return self._train_graph(ctx, loader_or_data)

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            data, idx_test = loader_or_data
            return GNNNodeEva(
                data,
                idx_test,
                ctx.gnn,
                ctx.answering,
                ctx.output_dim,
                ctx.device,
            )
        test_loader = loader_or_data
        return GNNGraphEva(
            test_loader,
            ctx.gnn,
            ctx.answering,
            ctx.output_dim,
            ctx.device,
        )

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _train_node(ctx: TaskContext, data_and_idx) -> float:
        data, train_idx = data_and_idx
        ctx.gnn.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()
        out = ctx.gnn(data.x, data.edge_index, batch=None)
        out = ctx.answering(out)
        loss = ctx.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    @staticmethod
    def _train_graph(ctx: TaskContext, train_loader) -> float:
        ctx.gnn.train()
        total_loss = 0.0
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            out = ctx.gnn(batch.x, batch.edge_index, batch.batch)
            out = ctx.answering(out)
            loss = ctx.criterion(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
