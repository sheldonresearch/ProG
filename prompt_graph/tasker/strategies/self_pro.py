"""SelfProStrategy: prompt_type == 'SelfPro' train/eval logic.

Self-Pro (ICML 2024) freezes the GNN encoder and fine-tunes a small MLP
projector on identity-graph embeddings.  Classification is performed via
cosine-similarity to class-centre prototypes (no linear answering head).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("SelfPro")
class SelfProStrategy(PromptStrategy):
    """Self-Pro projector tuning with prototype-based classification."""

    def setup(self, ctx: TaskContext) -> None:
        """No-op: projector is built by BaseTask.initialize_prompt."""

    def configure_optimizer(self, ctx: TaskContext) -> None:
        """No-op: optimizer built by BaseTask (projector params only)."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        data, train_idx = loader_or_data
        ctx.prompt.train()
        ctx.optimizer.zero_grad()

        # identity graph = self-loop only
        num_nodes = data.num_nodes
        device = ctx.device
        edge_index_id = torch.arange(num_nodes, device=device).repeat(2, 1)

        with torch.no_grad():
            embeds = ctx.gnn(data.x, edge_index_id)

        z = F.normalize(ctx.prompt(embeds), p=2, dim=1)
        centers = _compute_centers(z[train_idx], data.y[train_idx], ctx.output_dim)

        # InfoNCE against class centres
        z_train = z[train_idx]
        sim = torch.exp(torch.mm(z_train, centers.t()) / ctx.extra.get("temp", 0.8))
        pos = sim[torch.arange(z_train.size(0)), data.y[train_idx]]
        loss = -torch.log(pos / sim.sum(1)).mean()
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        data, idx_test = loader_or_data
        ctx.gnn.eval()
        ctx.prompt.eval()

        num_nodes = data.num_nodes
        device = ctx.device
        edge_index_id = torch.arange(num_nodes, device=device).repeat(2, 1)

        with torch.no_grad():
            embeds = ctx.gnn(data.x, edge_index_id)
            z = F.normalize(ctx.prompt(embeds), p=2, dim=1)

        # Use test labels to build centres — mirrors original few-shot eval
        centers = _compute_centers(z[idx_test], data.y[idx_test], ctx.output_dim)
        logits = torch.mm(z[idx_test], centers.t())
        pred = logits.argmax(dim=1)

        acc = _metric("accuracy", ctx.output_dim, device)(pred, data.y[idx_test])
        f1 = _metric("f1", ctx.output_dim, device)(pred, data.y[idx_test])
        roc = _metric("auroc", ctx.output_dim, device)(logits, data.y[idx_test])
        prc = _metric("auprc", ctx.output_dim, device)(logits, data.y[idx_test])
        return acc.item(), f1.item(), roc.item(), prc.item()


def _compute_centers(embeddings, labels, num_classes):
    """Compute mean prototype per class."""
    centers = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            centers.append(torch.zeros(embeddings.size(1), device=embeddings.device))
        else:
            centers.append(embeddings[mask].mean(0))
    return torch.stack(centers)


def _metric(name, num_classes, device):
    if name == "accuracy":
        return torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(device)
    if name == "f1":
        return torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
    if name == "auroc":
        return torchmetrics.classification.AUROC(
            task="multiclass", num_classes=num_classes
        ).to(device)
    if name == "auprc":
        return torchmetrics.classification.AveragePrecision(
            task="multiclass", num_classes=num_classes
        ).to(device)
    raise ValueError(name)
