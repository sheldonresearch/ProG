"""GraphPrompterStrategy: prompt_type == 'GraphPrompter' train/eval logic.

GraphPrompter (KDD 2025) uses metagraph propagation with adaptive prompt
selection (kNN + scoring MLP + LFU cache).  For graph-level tasks the GNN
backbone returns logits directly (readout + decode are handled inside the
prompt module); for node-level tasks the prompt returns enhanced embeddings
and the standard answering head is applied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prompt_graph.evaluation import GNNNodeEva

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("GraphPrompter")
class GraphPrompterStrategy(PromptStrategy):
    """Adaptive prompt selection via metagraph propagation."""

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
        return self._eval_graph(ctx, loader_or_data)

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
        total_loss = 0.0
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            # GraphPrompter returns logits directly for graph-level tasks
            out = ctx.gnn(
                batch.x,
                batch.edge_index,
                batch.batch,
                prompt=ctx.prompt,
                prompt_type="GraphPrompter",
            )
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @staticmethod
    def _eval_graph(ctx: TaskContext, test_loader) -> tuple:
        ctx.prompt.eval()
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(ctx.device)
                out = ctx.gnn(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    prompt=ctx.prompt,
                    prompt_type="GraphPrompter",
                )
                pred = out.argmax(dim=1)
                pred_labels.extend(pred.cpu().tolist())
                true_labels.extend(batch.y.cpu().tolist())

        import numpy as np
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)

        # Handle binary / multi-class ROC-AUC
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
                    np.eye(n_classes)[true_labels], np.eye(n_classes)[pred_labels], average="macro"
                )
            except ValueError:
                roc = 0.0
                prc = 0.0

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="macro")
        return acc, f1, roc, prc
