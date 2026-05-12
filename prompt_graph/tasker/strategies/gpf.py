"""GPFStrategy and GPFPlusStrategy: prompt-tuning via feature-vector prompts.

Encapsulates the behaviour previously inlined in ``NodeTask.GPFTrain`` /
``GraphTask.GPFTrain`` and the matching ``GPFEva`` calls. The two prompts
share a single train/eval loop differing only in how ``prompt`` is built.

Reads ``ctx.extra['input_dim']`` to construct the prompt module in ``setup``.
"""
from __future__ import annotations

from prompt_graph.evaluation import GPFEva
from prompt_graph.prompt import GPF, GPF_plus

from ..strategy import PromptStrategy, TaskContext, register_strategy


class _GPFBaseStrategy(PromptStrategy):
    """Shared train/eval loop for GPF-family prompts."""

    def train_epoch(self, ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        total_loss = 0.0
        for batch in train_loader:
            ctx.optimizer.zero_grad()
            batch = batch.to(ctx.device)
            batch.x = ctx.prompt.add(batch.x)
            out = ctx.gnn(
                batch.x,
                batch.edge_index,
                batch.batch,
                prompt=ctx.prompt,
                prompt_type=self.name,
            )
            out = ctx.answering(out)
            loss = ctx.criterion(out, batch.y)
            loss.backward()
            ctx.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        return GPFEva(
            test_loader,
            ctx.gnn,
            ctx.prompt,
            ctx.answering,
            ctx.output_dim,
            ctx.device,
        )


@register_strategy('GPF')
class GPFStrategy(_GPFBaseStrategy):
    def setup(self, ctx: TaskContext) -> None:
        ctx.prompt = GPF(ctx.extra['input_dim']).to(ctx.device)


@register_strategy('GPF-plus')
class GPFPlusStrategy(_GPFBaseStrategy):
    def setup(self, ctx: TaskContext) -> None:
        ctx.prompt = GPF_plus(ctx.extra['input_dim'], 20).to(ctx.device)
