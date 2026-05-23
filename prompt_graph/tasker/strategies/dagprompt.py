"""DAGPrompTStrategy: prompt_type == 'DAGPrompT' train/eval logic.

DAGPrompT (NeurIPS 2024) learns multi-hop re-weighting prompts and
parameterised class centres.  The GNN returns a stack of graph-level
embeddings per hop; loss is summed across hops.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prompt_graph.prompt.DAGPrompT import center_embedding_multihop

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
        num_batches = 0

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
            # Compute empirical per-hop class centres from this batch, then add
            # the learnable residual produced by ParameterizedMultiHopCenterEmbedding.
            empirical_centers = []
            class_counts = []
            for hop_emb in out_embeddings:
                c, cnt = center_embedding_multihop(hop_emb, batch.y, ctx.output_dim)
                empirical_centers.append(c)
                class_counts.append(cnt)
            centers = ctx.param_center_embeddings(empirical_centers)

            if accumulated_centers is None:
                accumulated_centers = centers
                accumulated_counts = class_counts
            else:
                accumulated_centers = [
                    acc + c * cnt.unsqueeze(-1)
                    for acc, c, cnt in zip(accumulated_centers, centers, class_counts)
                ]
                accumulated_counts = [
                    acc + cnt for acc, cnt in zip(accumulated_counts, class_counts)
                ]
            loss = _dagprompt_loss(out_embeddings, centers, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        self.mean_centers = [
            c / (cnt.unsqueeze(-1) + 1e-8)
            for c, cnt in zip(accumulated_centers, accumulated_counts)
        ]
        return total_loss / max(num_batches, 1)

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        ctx.gnn.eval()
        ctx.prompt.eval()
        device = ctx.device
        pred_labels = []
        true_labels = []
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
                    sim = F.cosine_similarity(
                        emb.unsqueeze(1), self.mean_centers[i].unsqueeze(0), dim=-1
                    )
                    if sim_accum is None:
                        sim_accum = sim * ctx.prompt.gamma[i]
                    else:
                        sim_accum = sim_accum + sim * ctx.prompt.gamma[i]
                pred = sim_accum.argmax(dim=1)
                pred_labels.extend(pred.cpu().tolist())
                true_labels.extend(batch.y.cpu().tolist())

        # Compute acc / macro-F1 / ROC-AUC / AUPRC. Mirrors the pattern in
        # GraphPrompterStrategy._eval_graph: try/except on degenerate cases
        # (only one class present, etc.) and fall back to 0.0.
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            roc_auc_score,
        )

        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)

        n_classes = ctx.output_dim
        if n_classes == 2:
            try:
                roc = roc_auc_score(true_labels, pred_labels)
                prc = average_precision_score(true_labels, pred_labels)
            except ValueError:
                roc = 0.0
                prc = 0.0
        else:
            try:
                roc = roc_auc_score(true_labels, pred_labels, multi_class="ovr", average="macro")
                prc = average_precision_score(
                    np.eye(n_classes)[true_labels],
                    np.eye(n_classes)[pred_labels],
                    average="macro",
                )
            except ValueError:
                roc = 0.0
                prc = 0.0

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="macro")
        return acc, f1, roc, prc


def _dagprompt_loss(embeddings, centers, labels, tau=0.1):
    total_loss = 0.0
    for emb, center in zip(embeddings, centers):
        sim = F.cosine_similarity(emb.unsqueeze(1), center.unsqueeze(0), dim=-1) / tau
        exp_sim = torch.exp(sim)
        pos = exp_sim.gather(1, labels.view(-1, 1))
        loss = -torch.log(pos / exp_sim.sum(1, keepdim=True))
        total_loss += loss.sum()
    return total_loss
