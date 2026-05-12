"""MultiGpromptStrategy: prompt_type == 'MultiGprompt' train/eval logic.

The most strongly coupled strategy in Phase 4: it relies on
``NodePrePrompt`` (pretrain/MultiGPrompt.py), ``featureprompt`` /
``downprompt`` (prompt/MultiGprompt.py), and the sparse adjacency
matrix + dense feature tensor produced by
``NodeTask.load_multigprompt_data``.

Only NodeTask is supported — there is no GraphTask path for MultiGprompt.

Loss-for-loss parity with the legacy ``NodeTask.MultiGpromptTrain`` /
``MultiGpromptEva`` branch is preserved, including the unusual
``loss.backward(retain_graph=True)`` call (the only place in the codebase
that retains the graph; needed because ``feature_prompt(features)`` shares
parameters across epochs).

``pretrain_embs`` / ``test_embs`` are produced once per fold on the
orchestrator side (``NodeTask.run`` precomputes them) and threaded in via
``ctx.extra`` so the strategy itself remains stateless.
"""

from __future__ import annotations

from prompt_graph.evaluation import MultiGpromptEva

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy("MultiGprompt")
class MultiGpromptStrategy(PromptStrategy):
    """MultiGprompt prompt-tuning train/eval (NodeTask only)."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        train_lbls, idx_train = loader_or_data
        down_prompt = ctx.extra["DownPrompt"]
        feature_prompt = ctx.extra["feature_prompt"]
        preprompt = ctx.extra["Preprompt"]
        features = ctx.extra["features"]
        sp_adj = ctx.extra["sp_adj"]
        pretrain_embs = ctx.extra["pretrain_embs"]

        down_prompt.train()
        ctx.optimizer.zero_grad()
        prompt_feature = feature_prompt(features)
        embeds1 = preprompt.gcn(prompt_feature, sp_adj, True, False)
        pretrain_embs1 = embeds1[0, idx_train]
        logits = (
            down_prompt(
                pretrain_embs,
                pretrain_embs1,
                train_lbls,
                1,
            )
            .float()
            .to(ctx.device)
        )
        loss = ctx.criterion(logits, train_lbls)
        loss.backward(retain_graph=True)
        ctx.optimizer.step()
        return loss.item()

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        test_lbls, idx_test = loader_or_data
        feature_prompt = ctx.extra["feature_prompt"]
        features = ctx.extra["features"]
        prompt_feature = feature_prompt(features)
        return MultiGpromptEva(
            ctx.extra["test_embs"],
            test_lbls,
            idx_test,
            prompt_feature,
            ctx.extra["Preprompt"],
            ctx.extra["DownPrompt"],
            ctx.extra["sp_adj"],
            ctx.output_dim,
            ctx.device,
        )
