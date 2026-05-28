"""GraphPrompter full model (KDD 2025).

Integrates supernode aggregation, metagraph propagation, adaptive prompt
selection (kNN + scoring MLP), and LFU caching into a single module that can
be used as a drop-in replacement for the standard ``answering`` head in ProG.

The model receives node embeddings from the GNN backbone, pools them into
supernode representations, propagates through a metagraph of supernodes +
learnable label embeddings, and decodes via cosine-similarity logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import LFUCacheE
from .layers import BgGraphToSupernodePropagator, MetaGNN, SupernodeToBgGraphPropagator


class GraphPrompterModel(nn.Module):
    """Full GraphPrompter model for graph / node classification.

    Args:
        emb_dim: Embedding dimension (must match GNN hidden dim).
        num_classes: Number of output classes.
        edge_attr_dim: Dimension of metagraph edge attributes (default 2).
        num_prompts: Number of candidate prompt supernodes per class.
        shots: Support-set size (used for kNN selection scaling).
        temp: Temperature for kNN top-k scaling.
        select_lambda: Weight for select_layers in kNN scoring.
        use_knn: Whether to enable kNN prompt selection.
        use_select: Whether to use the scoring MLP.
        use_cache: Whether to enable LFU caching.
        cache_cap: LFU cache capacity per class.
        meta_layers: Number of metagraph GNN layers.
        meta_heads: Number of attention heads in metagraph GNN.
        skip_path: If True, use residual connections.
        task_type: ``"graph"`` or ``"node"``.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        edge_attr_dim: int = 2,
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
        skip_path: bool = False,
        task_type: str = "graph",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.shots = shots
        self.temp = temp
        self.select_lambda = select_lambda
        self.use_knn = use_knn
        self.use_select = use_select
        self.use_cache = use_cache
        self.skip_path = skip_path
        self.task_type = task_type

        # Learnable label / prompt embeddings
        self.learned_label_embedding = nn.Embedding(num_classes, emb_dim)
        self.prompts = nn.Parameter(torch.randn(num_classes, num_prompts, emb_dim))

        # Input / output projections
        self.initial_input_mlp = nn.Linear(emb_dim, emb_dim)
        self.initial_label_mlp = nn.Linear(emb_dim, emb_dim)
        self.final_input_mlp = nn.Linear(emb_dim, emb_dim)
        self.final_label_mlp = nn.Linear(emb_dim, emb_dim)

        # Prompt selection MLP
        self.select_layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

        # Supernode aggregation (graph-level → one supernode per graph)
        self.supernode_aggr = BgGraphToSupernodePropagator(aggr="mean")
        self.supernode_to_bg = SupernodeToBgGraphPropagator(emb_dim)

        # Metagraph GNN
        self.meta_gnn = MetaGNN(
            edge_attr_dim=edge_attr_dim,
            emb_dim=emb_dim,
            heads=meta_heads,
            n_layers=meta_layers,
            dropout=0.0,
        )

        # Cosine similarity decoder
        self.cos = nn.CosineSimilarity(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0)

        # LFU cache
        self.cache_strategy = (
            LFUCacheE(capacity=cache_cap * num_classes, ways=num_classes) if use_cache else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompts)
        nn.init.xavier_uniform_(self.learned_label_embedding.weight)
        for m in [
            self.initial_input_mlp,
            self.initial_label_mlp,
            self.final_input_mlp,
            self.final_label_mlp,
        ]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for layer in self.select_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def decode(
        self,
        x_input: torch.Tensor,
        x_label: torch.Tensor,
        metagraph_edge_index: torch.Tensor,
        edgelist_bipartite: bool = True,
    ) -> torch.Tensor:
        """Cosine-similarity decoder.

        Returns logits of shape ``[num_input_nodes, num_labels]``.
        """
        x_input = F.normalize(x_input, p=2, dim=-1)
        x_label = F.normalize(x_label, p=2, dim=-1)
        if edgelist_bipartite:
            src = metagraph_edge_index[0]
            tgt = metagraph_edge_index[1]
            logits = self.cos(x_input[src], x_label[tgt - x_input.shape[0]])
            # Reconstruct dense matrix from bipartite edges
            # edge attr pattern: [0,0] = support-support, [0,1] = support-query, etc.
            # Simplified: just compute full cosine matrix
            logits = self.logit_scale * torch.mm(x_input, x_label.t())
        else:
            logits = self.logit_scale * torch.mm(x_input, x_label.t())
        return logits

    def _build_metagraph_edges(
        self,
        num_supernodes: int,
        num_labels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a complete bipartite edge index + attributes.

        Edge attributes: ``[0, 1]`` for support→label, ``[0, -1]`` for
        negative / query connections (simplified).
        """
        # Complete bipartite: every supernode connected to every label
        src = torch.arange(num_supernodes, device=device).repeat_interleave(num_labels)
        tgt = torch.arange(num_labels, device=device).repeat(num_supernodes)
        edge_index = torch.stack([src, tgt + num_supernodes], dim=0)
        # Edge attr: [0, 1] for all edges (simplified)
        edge_attr = torch.tensor([[0.0, 1.0]], device=device).repeat(edge_index.shape[1], 1)
        return edge_index, edge_attr

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        supernode_edge_index: torch.Tensor | None = None,
        supernode_idx: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_emb: Node embeddings from GNN backbone ``[N, emb_dim]``.
            batch: Batch vector for graph-level tasks ``[N]``.
            edge_index: Background graph edge index (for supernode propagation).
            supernode_edge_index: Edges from bg nodes to supernodes ``[2, E]``.
            supernode_idx: Indices of supernodes in the node list.
            y: Ground-truth labels (for cache updates).
            training: Training mode flag.

        Returns:
            Logits ``[num_queries, num_classes]``.
        """
        device = node_emb.device
        num_labels = self.num_classes

        # ------------------------------------------------------------------
        # 1. Supernode aggregation
        # ------------------------------------------------------------------
        if self.task_type == "graph" and batch is not None:
            # Graph-level: one supernode per graph via pooling
            # Build pseudo supernode_edge_index from batch vector
            num_graphs = int(batch.max().item()) + 1
            # Each node connects to its graph supernode
            src = torch.arange(node_emb.shape[0], device=device)
            tgt = batch.clone()
            supernode_edge_index = torch.stack([src, tgt], dim=0)
            supernode_idx = torch.arange(num_graphs, device=device)
            x_input = self.supernode_aggr(node_emb, supernode_edge_index, supernode_idx)
        elif supernode_edge_index is not None and supernode_idx is not None:
            x_input = self.supernode_aggr(node_emb, supernode_edge_index, supernode_idx)
        else:
            # Fallback: treat every node as its own supernode
            x_input = node_emb
            supernode_idx = torch.arange(node_emb.shape[0], device=device)

        num_supernodes = x_input.shape[0]
        # Remember the actual supernode count so that, after kNN augmentation
        # (which prepends selected prompts to ``x_input`` below), we can slice
        # logits back down to one row per real supernode. Without this slice,
        # the returned logits include rows for the kNN candidate prompts and
        # the batch_size of ``logits`` would no longer match ``batch.y`` --
        # the strategy's cross_entropy then raises
        # ``ValueError: input batch_size (X) to match target batch_size (Y)``.
        num_actual_supernodes = num_supernodes

        # ------------------------------------------------------------------
        # 2. Initialise label embeddings
        # ------------------------------------------------------------------
        x_label = self.learned_label_embedding(torch.arange(num_labels, device=device))
        x_label = self.initial_label_mlp(x_label)

        # ------------------------------------------------------------------
        # 3. kNN prompt selection (adaptive)
        # ------------------------------------------------------------------
        if self.use_knn and num_supernodes > num_labels:
            # Use the learnable prompts as candidate pool
            # prompts: [num_classes, num_prompts, emb_dim]
            prompt_embs = self.prompts.view(-1, self.emb_dim)  # [num_classes*num_prompts, emb_dim]
            n_k = max(1, (self.shots // 10) * self.temp if (self.shots // 10) > 1 else self.temp)
            n_k = int(n_k)

            selected_prompts = []
            for i in range(num_supernodes):
                sim = self.cos(x_input[i].unsqueeze(0), prompt_embs)  # [num_prompts_total]
                if self.use_select:
                    prompt_weights = self.sigmoid(self.select_layers(prompt_embs)).squeeze(-1)
                    sim = sim + prompt_weights * self.select_lambda
                topk_val, topk_idx = torch.topk(sim, k=min(n_k, prompt_embs.shape[0]))
                selected_prompts.append(prompt_embs[topk_idx])

            if selected_prompts:
                selected_prompts = torch.cat(
                    selected_prompts, dim=0
                )  # [num_supernodes*n_k, emb_dim]
                # Rebuild x_input to include selected prompts + original supernodes
                x_input = torch.cat([selected_prompts, x_input], dim=0)
                num_supernodes = x_input.shape[0]

        # ------------------------------------------------------------------
        # 4. Select-layer weighting
        # ------------------------------------------------------------------
        if self.use_select:
            x_weights = self.sigmoid(self.select_layers(x_input))
            x_input = x_input * x_weights

        # ------------------------------------------------------------------
        # 5. Build metagraph and propagate
        # ------------------------------------------------------------------
        metagraph_edge_index, metagraph_edge_attr = self._build_metagraph_edges(
            num_supernodes, num_labels, device
        )

        # Concatenate supernodes + labels as metagraph nodes
        metagraph_x = torch.cat([x_input, x_label], dim=0)
        # In supervised mode all supernode->label edges are treated as query edges
        query_mask = torch.ones(metagraph_edge_index.shape[1], dtype=torch.bool, device=device)

        metagraph_x = self.meta_gnn(
            metagraph_x,
            metagraph_edge_index,
            metagraph_edge_attr,
            query_mask,
        )

        x_input = metagraph_x[:num_supernodes]
        x_label = metagraph_x[num_supernodes:]

        # ------------------------------------------------------------------
        # 6. Final projections + decode
        # ------------------------------------------------------------------
        x_input = self.final_input_mlp(x_input)
        x_label = self.final_label_mlp(x_label)

        logits = self.decode(x_input, x_label, metagraph_edge_index, edgelist_bipartite=False)

        # Slice out logits for the original supernodes only (drop kNN-augmented
        # prompt rows). ``x_input`` ordering after kNN is
        # ``[selected_prompts, original_supernodes]`` so the real rows are the
        # last ``num_actual_supernodes`` of the leading-supernode block.
        if logits.shape[0] >= num_actual_supernodes:
            # ``decode`` may return either [num_supernodes, num_labels] or
            # [num_supernodes * num_labels] depending on edgelist_bipartite=False
            # path; here it's the matrix form -- take the trailing real rows.
            logits = logits[num_supernodes - num_actual_supernodes : num_supernodes]

        # ------------------------------------------------------------------
        # 7. Cache update (training only)
        # ------------------------------------------------------------------
        if training and self.use_cache and y is not None and self.cache_strategy is not None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                for i in range(logits.shape[0]):
                    self.cache_strategy.put(x_input[i].cpu(), preds[i].item())

        return logits
