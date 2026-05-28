"""ProdigyStrategy: prompt_type == 'Prodigy' train/eval logic.

Minimal port of the PRODIGY metagraph + supernode propagation idea into
ProG's strategy framework.  Prodigy acts as a **graph contextualizer**: it
receives node embeddings from the backbone GNN, builds a lightweight
supernode / metagraph structure, and returns richer contextualized embeddings.

* NodeTask – applied after the GNN (like GPPT).
* GraphTask – applied inside the GNN forward, just before readout (like Gprompt).
"""

from __future__ import annotations

import torch
import torchmetrics

from prompt_graph.utils import get_logger

from ..strategy import PromptStrategy, TaskContext, register_strategy

logger = get_logger(__name__)


@register_strategy("Prodigy")
class ProdigyStrategy(PromptStrategy):
    """Prodigy in-context graph prompt."""

    def setup(self, ctx: TaskContext) -> None:
        """Ensure answering head exists (GraphTask path needs it)."""
        if ctx.answering is None:
            from torch import nn

            ctx.answering = nn.Sequential(
                nn.Linear(ctx.hid_dim, ctx.output_dim),
                nn.Softmax(dim=1),
            ).to(ctx.device)

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """Adam over gnn + prompt + answering params."""
        from torch import optim

        lr = ctx.extra.get("lr", 1e-3)
        wd = ctx.extra.get("wd", 5e-4)
        param_groups = [
            {"params": ctx.gnn.parameters()},
            {"params": ctx.prompt.parameters()},
            {"params": ctx.answering.parameters()},
        ]
        ctx.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=wd)

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            data, train_idx = loader_or_data
            return self._train_node(ctx, data, train_idx)
        return self._train_graph(ctx, loader_or_data)

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        task_type = ctx.extra.get("task_type", "NodeTask")
        if task_type == "NodeTask":
            data, idx_test = loader_or_data
            return self._eval_node(ctx, data, idx_test)
        test_loader = loader_or_data
        return self._eval_graph(ctx, test_loader)

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _train_node(ctx: TaskContext, data, train_idx) -> float:
        ctx.gnn.train()
        ctx.prompt.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()
        node_embedding = ctx.gnn(data.x, data.edge_index)
        out = ctx.prompt(node_embedding, data.edge_index)
        out = ctx.answering(out)
        loss = ctx.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    @staticmethod
    def _train_graph(ctx: TaskContext, train_loader) -> float:
        ctx.gnn.train()
        ctx.prompt.train()
        ctx.answering.train()
        total_loss = 0.0
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            out = ctx.gnn(
                batch.x,
                batch.edge_index,
                batch.batch,
                prompt=ctx.prompt,
                prompt_type="Prodigy",
            )
            out = ctx.answering(out)
            loss = ctx.criterion(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @staticmethod
    def _eval_node(ctx: TaskContext, data, idx_test) -> tuple:
        ctx.gnn.eval()
        ctx.prompt.eval()
        ctx.answering.eval()
        device = ctx.device
        num_class = ctx.output_dim

        accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_class
        ).to(device)
        macro_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_class, average="macro"
        ).to(device)
        auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_class
        ).to(device)
        auprc = torchmetrics.classification.AveragePrecision(
            task="multiclass", num_classes=num_class
        ).to(device)

        with torch.no_grad():
            node_embedding = ctx.gnn(data.x, data.edge_index)
            out = ctx.prompt(node_embedding, data.edge_index)
            out = ctx.answering(out)
            pred = out.argmax(dim=1)

            acc = accuracy(pred[idx_test], data.y[idx_test])
            f1 = macro_f1(pred[idx_test], data.y[idx_test])
            roc = auroc(out[idx_test], data.y[idx_test])
            prc = auprc(out[idx_test], data.y[idx_test])

        return acc.item(), f1.item(), roc.item(), prc.item()

    @staticmethod
    def _eval_graph(ctx: TaskContext, test_loader) -> tuple:
        ctx.gnn.eval()
        ctx.prompt.eval()
        ctx.answering.eval()
        device = ctx.device
        num_class = ctx.output_dim

        accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_class
        ).to(device)
        macro_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_class, average="macro"
        ).to(device)
        auroc = torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_class
        ).to(device)
        auprc = torchmetrics.classification.AveragePrecision(
            task="multiclass", num_classes=num_class
        ).to(device)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = ctx.gnn(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    prompt=ctx.prompt,
                    prompt_type="Prodigy",
                )
                out = ctx.answering(out)
                pred = out.argmax(dim=1)
                accuracy(pred, batch.y)
                macro_f1(pred, batch.y)
                auroc(out, batch.y)
                auprc(out, batch.y)

        return (
            accuracy.compute().item(),
            macro_f1.compute().item(),
            auroc.compute().item(),
            auprc.compute().item(),
        )
