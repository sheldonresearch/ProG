"""ProNoGStrategy: prompt_type == 'ProNoG' train/eval logic.

ProNoG (KDD 2025) freezes the GNN and fine-tunes adaptive hop-specific
prompt weights + a meta-net.  Classification is prototype-based.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("ProNoG")
class ProNoGStrategy(PromptStrategy):
    """ProNoG neighbour-aggregation prompt tuning."""

    def setup(self, ctx: TaskContext) -> None:
        """Pre-compute node embeddings and neighbour lists."""
        data = ctx.extra["data"]
        device = ctx.device
        ctx.gnn.eval()
        with torch.no_grad():
            embeds = ctx.gnn(data.x, data.edge_index).detach()

        neighbors, neighbors_2hop = _build_neighbor_lists(data.edge_index, data.num_nodes, device)

        from prompt_graph.prompt.ProNoGPrompt import ProNoGPrompt

        ctx.prompt = ProNoGPrompt(
            embeds=embeds,
            neighbors=neighbors,
            neighbors_2hop=neighbors_2hop,
            nb_classes=ctx.output_dim,
            hidden_size=ctx.hid_dim,
            bottleneck_size=ctx.extra.get("bottleneck_size", 64),
            dropout=ctx.extra.get("dropout", 0.05),
            use_metanet=ctx.extra.get("use_metanet", 1),
            multi_prompt=ctx.extra.get("multi_prompt", 1),
        ).to(device)

        # Freeze GNN
        for p in ctx.gnn.parameters():
            p.requires_grad = False

    def configure_optimizer(self, ctx: TaskContext) -> None:
        lr = ctx.extra.get("lr", 1e-3)
        wd = ctx.extra.get("wd", 5e-4)
        ctx.optimizer = torch.optim.Adam(ctx.prompt.parameters(), lr=lr, weight_decay=wd)

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        data, train_idx = loader_or_data
        ctx.prompt.train()
        ctx.optimizer.zero_grad()
        logits = ctx.prompt(train_idx, train_labels=data.y[train_idx])
        loss = F.cross_entropy(logits, data.y[train_idx])
        loss.backward()
        ctx.optimizer.step()
        return loss.item()

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        data, idx_test = loader_or_data
        ctx.gnn.eval()
        ctx.prompt.eval()
        with torch.no_grad():
            logits = ctx.prompt(idx_test)
            pred = logits.argmax(dim=1)

        device = ctx.device
        acc = _metric("accuracy", ctx.output_dim, device)(pred, data.y[idx_test])
        f1 = _metric("f1", ctx.output_dim, device)(pred, data.y[idx_test])
        roc = _metric("auroc", ctx.output_dim, device)(logits, data.y[idx_test])
        prc = _metric("auprc", ctx.output_dim, device)(logits, data.y[idx_test])
        return acc.item(), f1.item(), roc.item(), prc.item()


def _build_neighbor_lists(edge_index, num_nodes, device):
    """Build 1-hop and 2-hop neighbour index lists for each node."""
    # 1-hop
    neighbors = [[] for _ in range(num_nodes)]
    src, dst = edge_index.cpu().tolist()
    for s, d in zip(src, dst):
        neighbors[s].append(d)

    # 2-hop
    neighbors_2hop = [[] for _ in range(num_nodes)]
    for node in range(num_nodes):
        seen = set(neighbors[node])
        two_hop = set()
        for nbr in neighbors[node]:
            for nbr2 in neighbors[nbr]:
                if nbr2 != node and nbr2 not in seen:
                    two_hop.add(nbr2)
        neighbors_2hop[node] = list(two_hop)

    # Convert to tensors on target device for fast indexing
    neighbors = [torch.tensor(n, dtype=torch.long, device=device) if len(n) else torch.zeros(0, dtype=torch.long, device=device) for n in neighbors]
    neighbors_2hop = [torch.tensor(n, dtype=torch.long, device=device) if len(n) else torch.zeros(0, dtype=torch.long, device=device) for n in neighbors_2hop]
    return neighbors, neighbors_2hop


def _metric(name, num_classes, device):
    if name == "accuracy":
        return torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    if name == "f1":
        return torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    if name == "auroc":
        return torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes).to(device)
    if name == "auprc":
        return torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
    raise ValueError(name)
