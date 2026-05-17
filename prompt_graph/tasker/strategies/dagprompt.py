"""DAGPrompTStrategy: prompt_type == 'DAGPrompT' train/eval logic.

DAGPrompT (NeurIPS 2024) learns multi-hop re-weighting prompts and
parameterised class centres.  The GNN returns a stack of graph-level
embeddings per hop; loss is summed across hops.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("DAGPrompT")
class DAGPrompTStrategy(PromptStrategy):
    """Multi-hop prototype prompt tuning."""

    def __init__(self):
        self.mean_centers = None

    def train_epoch(self, ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        ctx.gnn.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None

        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            out_embeddings = ctx.gnn(
                batch.x,
                batch.edge_index,
                batch.batch,
                prompt=ctx.prompt,
                prompt_type="DAGPrompT",
            )
            # out_embeddings: [hop_num, batch_size, hidden_dim]
            centers, class_counts = ctx.param_center_embeddings(out_embeddings, batch.y)
            if accumulated_centers is None:
                accumulated_centers = centers
                accumulated_counts = class_counts
            else:
                accumulated_centers = [
                    acc + c * cnt for acc, c, cnt in zip(accumulated_centers, centers, class_counts)
                ]
                accumulated_counts = [
                    acc + cnt for acc, cnt in zip(accumulated_counts, class_counts)
                ]
            loss = _dagprompt_loss(out_embeddings, centers, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()

        self.mean_centers = [c / (cnt + 1e-8) for c, cnt in zip(accumulated_centers, accumulated_counts)]
        return total_loss / len(train_loader)

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        ctx.gnn.eval()
        ctx.prompt.eval()
        device = ctx.device
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out_embeddings = ctx.gnn(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    prompt=ctx.prompt,
                    prompt_type="DAGPrompT",
                )
                # Accumulate similarity weighted by prompt gamma
                sim_accum = None
                for i, emb in enumerate(out_embeddings):
                    sim = F.cosine_similarity(emb.unsqueeze(1), self.mean_centers[i].unsqueeze(0), dim=-1)
                    if sim_accum is None:
                        sim_accum = sim * ctx.prompt.gamma[i]
                    else:
                        sim_accum = sim_accum + sim * ctx.prompt.gamma[i]
                pred = sim_accum.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        acc = correct / total if total > 0 else 0.0
        # F1 / ROC / PRC are not computed in the original DAGPrompT evaluator;
        # return placeholders for ProG compatibility.
        return acc, 0.0, 0.0, 0.0


def _dagprompt_loss(embeddings, centers, labels, tau=0.1):
    total_loss = 0.0
    for emb, center in zip(embeddings, centers):
        sim = F.cosine_similarity(emb.unsqueeze(1), center.unsqueeze(0), dim=-1) / tau
        exp_sim = torch.exp(sim)
        pos = exp_sim.gather(1, labels.view(-1, 1))
        loss = -torch.log(pos / exp_sim.sum(1, keepdim=True))
        total_loss += loss.sum()
    return total_loss
