"""AllInOneStrategy: prompt_type == 'All-in-one' alternating train/eval logic.

Encapsulates ``NodeTask.AllInOneTrain`` / ``GraphTask.AllInOneTrain`` and the
matching ``AllInOneEva`` call. Train alternates between tuning ``answering``
(frozen prompt) and tuning ``prompt`` (frozen answering) for the configured
inner-loop budgets.

Two preserved asymmetries vs. baseline:
  * NodeTask returns ``answer_loss``; GraphTask returns ``pg_loss``.
  * NodeTask also calls ``gnn.eval()`` before the answering phase;
    GraphTask does not. ``ctx.extra['task_type']`` selects.
"""

from __future__ import annotations

from prompt_graph.evaluation import AllInOneEva
from prompt_graph.utils import get_logger

from ..strategy import PromptStrategy, TaskContext, register_strategy

logger = get_logger(__name__)


@register_strategy("All-in-one")
class AllInOneStrategy(PromptStrategy):
    """Strategy for alternating prompt/answering tuning."""

    def train_epoch(self, ctx: TaskContext, train_loader) -> float:
        task_type = ctx.extra.get("task_type", "NodeTask")
        answer_epoch = ctx.extra.get("answer_epoch", 1)
        prompt_epoch = ctx.extra.get("prompt_epoch", 1)
        is_node_task = task_type == "NodeTask"

        ctx.answering.train()
        ctx.prompt.eval()
        if is_node_task:
            ctx.gnn.eval()
        answer_loss = None
        for epoch in range(1, answer_epoch + 1):
            answer_loss = ctx.prompt.Tune(
                train_loader,
                ctx.gnn,
                ctx.answering,
                ctx.criterion,
                ctx.answer_opi,
                ctx.device,
            )
            logger.info(
                "frozen gnn | frozen prompt | *tune answering function... "
                f"{epoch}/{answer_epoch} ,loss: {answer_loss:.4f} "
            )

        ctx.answering.eval()
        ctx.prompt.train()
        pg_loss = None
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = ctx.prompt.Tune(
                train_loader,
                ctx.gnn,
                ctx.answering,
                ctx.criterion,
                ctx.pg_opi,
                ctx.device,
            )
            logger.info(
                "frozen gnn | *tune prompt |frozen answering function... "
                f"{epoch}/{prompt_epoch} ,loss: {pg_loss:.4f} "
            )

        return answer_loss if is_node_task else pg_loss

    def evaluate(self, ctx: TaskContext, test_loader) -> tuple:
        return AllInOneEva(
            test_loader,
            ctx.prompt,
            ctx.gnn,
            ctx.answering,
            ctx.output_dim,
            ctx.device,
        )
