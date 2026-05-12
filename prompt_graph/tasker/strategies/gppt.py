"""GPPTStrategy: prompt_type == 'GPPT' train/eval logic.

Encapsulates ``NodeTask.GPPTtrain`` / ``GraphTask.GPPTtrain`` and the
matching ``GPPTEva`` / ``GPPTGraphEva`` calls. The strategy preserves
two GPPT-specific behaviours:

  * Loss has a ``+ 0.001 * constraint(...)`` task-token regularizer.
  * After each optimizer step, ``prompt.update_StructureToken_weight(
    prompt.get_mid_h())`` runs as out-of-optimizer state mutation.

``weight_init`` and the GraphTask test_loader reconstruction are kept on
the orchestrator side (NodeTask.run / GraphTask._gppt_weight_init) since
they depend on the few-shot folding logic. The strategy reads
``ctx.extra['task_type']`` to switch between node- and graph-level
training.
"""
from __future__ import annotations

import torch

from prompt_graph.evaluation import GPPTEva, GPPTGraphEva
from prompt_graph.utils import constraint

from ..strategy import PromptStrategy, TaskContext, register_strategy


@register_strategy('GPPT')
class GPPTStrategy(PromptStrategy):
    """GPPT prompt-tuning train/eval (NodeTask + GraphTask)."""

    def train_epoch(self, ctx: TaskContext, loader_or_data) -> float:
        task_type = ctx.extra.get('task_type', 'NodeTask')
        if task_type == 'NodeTask':
            data, train_idx = loader_or_data
            return self._train_node(ctx, data, train_idx)
        return self._train_graph(ctx, loader_or_data)

    def evaluate(self, ctx: TaskContext, loader_or_data) -> tuple:
        task_type = ctx.extra.get('task_type', 'NodeTask')
        if task_type == 'NodeTask':
            data, idx_test = loader_or_data
            return GPPTEva(data, idx_test, ctx.gnn, ctx.prompt, ctx.output_dim, ctx.device)
        test_loader = loader_or_data
        return GPPTGraphEva(test_loader, ctx.gnn, ctx.prompt, ctx.output_dim, ctx.device)

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _train_node(ctx: TaskContext, data, train_idx) -> float:
        ctx.prompt.train()
        node_embedding = ctx.gnn(data.x, data.edge_index)
        out = ctx.prompt(node_embedding, data.edge_index)
        loss = ctx.criterion(out[train_idx], data.y[train_idx])
        loss = loss + 0.001 * constraint(ctx.device, ctx.prompt.get_TaskToken())
        ctx.pg_opi.zero_grad()
        loss.backward()
        ctx.pg_opi.step()
        ctx.prompt.update_StructureToken_weight(ctx.prompt.get_mid_h())
        return loss.item()

    @staticmethod
    def _train_graph(ctx: TaskContext, train_loader) -> float:
        ctx.prompt.train()
        for batch in train_loader:
            temp_loss = torch.tensor(0.0, requires_grad=True).to(ctx.device)
            graph_list = batch.to_data_list()
            for index, graph in enumerate(graph_list):
                graph = graph.to(ctx.device)
                node_embedding = ctx.gnn(graph.x, graph.edge_index)
                out = ctx.prompt(node_embedding, graph.edge_index)
                loss = ctx.criterion(
                    out,
                    torch.full((1, graph.x.shape[0]), graph.y.item())
                    .reshape(-1).to(ctx.device),
                )
                temp_loss = temp_loss + loss + 0.001 * constraint(
                    ctx.device, ctx.prompt.get_TaskToken(),
                )
            temp_loss = temp_loss / (index + 1)
            ctx.pg_opi.zero_grad()
            temp_loss.backward()
            ctx.pg_opi.step()
            ctx.prompt.update_StructureToken_weight(ctx.prompt.get_mid_h())
        return temp_loss.item()
