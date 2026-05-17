"""GraphPrompterPrompt: adaptive prompt selection (KDD 2025) full version.

This module wraps :class:`GraphPrompterModel` and exposes a unified interface
compatible with ProG's ``gnn.py`` dispatch logic.

* **Graph-level** (``batch is not None``): returns per-graph logits directly.
* **Node-level** (``batch is None``): returns prompt-augmented node embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prompt_graph.prompt.GraphPrompter import GraphPrompterModel


class GraphPrompterPrompt(nn.Module):
    """Adaptive prompt selection via metagraph propagation.

    Args:
        emb_dim: Dimension of node / graph embeddings.
        num_classes: Number of target classes.
        num_prompts: Candidate prompts per class.
        shots: Support-set size (scales kNN top-k).
        temp: kNN temperature.
        select_lambda: Weight for scoring-MLP in kNN voting.
        use_knn: Enable adaptive kNN prompt selection.
        use_select: Enable scoring MLP.
        use_cache: Enable LFU embedding cache.
        cache_cap: Cache capacity per class.
        meta_layers: Number of metagraph GNN layers.
        meta_heads: Attention heads in metagraph GNN.
        task_type: ``"graph"`` or ``"node"``.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        num_prompts: int = 10,
        shots: int = 10,
        temp: float = 1.0,
        select_lambda: float = 0.5,
        use_knn: bool = True,
        use_select: bool = True,
        use_cache: bool = False,
        cache_cap: int = 5,
        meta_layers: int = 1,
        meta_heads: int = 4,
        task_type: str = "graph",
    ):
        super().__init__()
        self.model = GraphPrompterModel(
            emb_dim=emb_dim,
            num_classes=num_classes,
            num_prompts=num_prompts,
            shots=shots,
            temp=temp,
            select_lambda=select_lambda,
            use_knn=use_knn,
            use_select=use_select,
            use_cache=use_cache,
            cache_cap=cache_cap,
            meta_layers=meta_layers,
            meta_heads=meta_heads,
            task_type=task_type,
        )
        self.task_type = task_type

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GraphPrompter forward.

        Args:
            x: Node embeddings ``[N, emb_dim]``.
            batch: Batch vector for graph-level tasks.
            edge_index: Background graph edge index.
            y: Ground-truth labels (for cache updates).

        Returns:
            Graph-level → logits ``[B, num_classes]``.
            Node-level → enhanced embeddings ``[N, emb_dim]``.
        """
        if batch is not None or self.task_type == "graph":
            # Graph-level: return logits directly
            return self.model(
                node_emb=x,
                batch=batch,
                edge_index=edge_index,
                y=y,
                training=self.training,
            )
        # Node-level: return enhanced embeddings
        # Treat each node as its own supernode for metagraph propagation
        num_nodes = x.shape[0]
        device = x.device
        supernode_idx = torch.arange(num_nodes, device=device)
        # Build trivial supernode_edge_index (each node connects to itself)
        supernode_edge_index = torch.stack([
            torch.arange(num_nodes, device=device),
            torch.arange(num_nodes, device=device),
        ], dim=0)

        # Temporarily disable kNN for node-level (kNN changes supernode count)
        old_use_knn = self.model.use_knn
        self.model.use_knn = False
        logits = self.model(
            node_emb=x,
            edge_index=edge_index,
            supernode_edge_index=supernode_edge_index,
            supernode_idx=supernode_idx,
            y=y,
            training=self.training,
        )
        self.model.use_knn = old_use_knn
        # logits: [num_nodes, num_classes]
        # Convert back to embedding space via linear combination of label embeddings
        # This preserves the prompt-augmented signal while staying in emb_dim space
        label_embs = self.model.learned_label_embedding.weight  # [num_classes, emb_dim]
        probs = torch.softmax(logits, dim=-1)
        enhanced = torch.mm(probs, label_embs)  # [num_nodes, emb_dim]
        return x + enhanced
