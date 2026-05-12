"""GpromptStrategy: prompt_type == 'Gprompt' train/eval logic.

Encapsulates ``GpromptTrain`` + ``GpromptEva`` for both NodeTask and GraphTask.
Stateful: per-fold class-center embeddings produced during training must be
handed off to evaluation. The caller is expected to reuse the **same**
strategy instance across all ``train_epoch`` calls of a fold and the
matching ``evaluate`` call, so ``self.mean_centers`` survives the
train→eval transition.
"""
from __future__ import annotations

from prompt_graph.evaluation import GpromptEva
from prompt_graph.utils import Gprompt_tuning_loss, center_embedding

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy('Gprompt')
class GpromptStrategy(PromptStrategy):
    """Center-prototype prompt; uses ``pg_opi`` for optimization."""

    def __init__(self):
        self.mean_centers = None

    def train_epoch(self, ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        criterion = Gprompt_tuning_loss()

        for batch in train_loader:
            ctx.pg_opi.zero_grad()
            batch = batch.to(ctx.device)
            out = ctx.gnn(
                batch.x, batch.edge_index, batch.batch,
                prompt=ctx.prompt, prompt_type='Gprompt',
            )
            center, class_counts = center_embedding(out, batch.y, ctx.output_dim)
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers = accumulated_centers + center * class_counts
                accumulated_counts = accumulated_counts + class_counts
            loss = criterion(out, center, batch.y)
            loss.backward()
            ctx.pg_opi.step()
            total_loss += loss.item()

        self.mean_centers = accumulated_centers / accumulated_counts
        return total_loss / len(train_loader)

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        return GpromptEva(
            test_loader, ctx.gnn, ctx.prompt,
            self.mean_centers, ctx.output_dim, ctx.device,
        )
