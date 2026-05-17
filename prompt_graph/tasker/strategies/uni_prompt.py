"""UniPromptStrategy: prompt_type == 'UniPrompt' train/eval logic.

UniPrompt (NeurIPS 2025) builds a k-NN prompt graph and fuses it with the
original graph before each GNN forward pass.  Because ProG's standard conv
registry only guarantees ``edge_weight`` support for ``GCNConv``, this
strategy is most reliable when the backbone is ``conv_type='GCN'``.
"""

from __future__ import annotations

import torch
import torchmetrics
from torch_geometric.utils import add_self_loops, degree

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("UniPrompt")
class UniPromptStrategy(PromptStrategy):
    """Universal graph adaptation via k-NN edge prompting."""

    def setup(self, ctx: TaskContext) -> None:
        """No-op: prompt / answering / optimizer are built by BaseTask."""

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
            return _eval_node(ctx, data, idx_test)
        return _eval_graph(ctx, loader_or_data)

    @staticmethod
    def _train_node(ctx: TaskContext, data_and_idx) -> float:
        data, train_idx = data_and_idx
        ctx.prompt.train()
        ctx.answering.train()
        ctx.optimizer.zero_grad()
        embeds = _fuse_and_embed(ctx, data, ctx.prompt, ctx.extra.get("tau", 0.99))
        out = ctx.answering(embeds)
        loss = ctx.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    @staticmethod
    def _train_graph(ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        ctx.answering.train()
        total_loss = 0.0
        tau = ctx.extra.get("tau", 0.99)
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            embeds = _fuse_and_embed(ctx, batch, ctx.prompt, tau, batch=batch.batch)
            out = ctx.answering(embeds)
            loss = ctx.criterion(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


def _normalize_edge_index(edge_index, num_nodes, device):
    """Compute symmetrically-normalised edge weights for GCN."""
    edge_weight = torch.ones(
        edge_index.size(1), dtype=torch.float32, device=device
    )
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes
    )
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=edge_weight.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    return edge_index, deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight


def _fuse_and_embed(ctx, data, prompt, tau, batch=None):
    """Fuse original + prompt edges and run GNN."""
    device = ctx.device
    num_nodes = data.num_nodes
    orig_index, orig_weight = _normalize_edge_index(
        data.edge_index, num_nodes, device
    )
    pt_index, pt_weight = prompt()
    fused_index, fused_weight = prompt.edge_fuse(
        orig_index, orig_weight, pt_index, pt_weight, tau
    )
    if batch is not None:
        embeds = ctx.gnn(data.x, fused_index, batch=batch, edge_weight=fused_weight)
    else:
        embeds = ctx.gnn(data.x, fused_index, edge_weight=fused_weight)
    return embeds


def _eval_node(ctx, data, idx_test):
    """Node-level evaluation with fused edges."""
    ctx.gnn.eval()
    ctx.answering.eval()
    accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)
    macro_f1 = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=ctx.output_dim, average="macro"
    ).to(ctx.device)
    auroc = torchmetrics.classification.AUROC(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)
    auprc = torchmetrics.classification.AveragePrecision(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)

    with torch.no_grad():
        embeds = _fuse_and_embed(ctx, data, ctx.prompt, ctx.extra.get("tau", 0.99))
        out = ctx.answering(embeds)
        pred = out.argmax(dim=1)
        acc = accuracy(pred[idx_test], data.y[idx_test])
        f1 = macro_f1(pred[idx_test], data.y[idx_test])
        roc = auroc(out[idx_test], data.y[idx_test])
        prc = auprc(out[idx_test], data.y[idx_test])
    return acc.item(), f1.item(), roc.item(), prc.item()


def _eval_graph(ctx, loader):
    """Graph-level evaluation with fused edges."""
    ctx.gnn.eval()
    ctx.answering.eval()
    accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)
    macro_f1 = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=ctx.output_dim, average="macro"
    ).to(ctx.device)
    auroc = torchmetrics.classification.AUROC(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)
    auprc = torchmetrics.classification.AveragePrecision(
        task="multiclass", num_classes=ctx.output_dim
    ).to(ctx.device)

    tau = ctx.extra.get("tau", 0.99)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(ctx.device)
            embeds = _fuse_and_embed(ctx, batch, ctx.prompt, tau, batch=batch.batch)
            out = ctx.answering(embeds)
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
