"""LinkTask — link-prediction downstream task with prompt-strategy integration.

Mirrors ``NodeTask`` / ``GraphTask``: subclasses :class:`BaseTask`, ``run()``
returns the same 9-tuple ``(avg_loss, acc_mean, acc_std, f1_mean, f1_std,
roc_mean, roc_std, prc_mean, prc_std)`` so ``bench.do_config_bench`` is
shape-compatible.

Architectural deltas vs Node/GraphTask
--------------------------------------
* **LinkTask owns the train/eval loop.** Where Node/GraphTask delegate to
  ``strategy.train_epoch`` / ``strategy.evaluate``, LinkTask runs its own
  BCE-with-logits + dot-product-decoder loop and dispatches to a small
  prompt-aware node encoder (:func:`encode_nodes_for_link`). This avoids
  forcing every existing strategy to grow a LinkTask branch.
* **Three explicit edge sets** (per
  :class:`torch_geometric.transforms.RandomLinkSplit` contract):
    1. *Message-passing graph*: ``data.edge_index`` — val + test positives
       are removed. For the few-shot path, the message-passing graph is the
       transductive train graph (full train positives) and only the
       *supervision* labels are subsampled to ``k_pos`` edges.
    2. *Supervision labels*: ``data.edge_label_index`` + ``data.edge_label``.
    3. *Eval labels*: same structure on the ``val_data`` / ``test_data``
       objects returned by ``RandomLinkSplit``.
* **Negative sampling at training time** uses the union of train+val+test
  positives as the exclusion mask so val/test positives can never be sampled
  as training negatives.
* **Output dim is binary** (``LINK_OUTPUT_DIM = 2``) regardless of what the
  underlying loader returned, because the link decoder is per-edge binary.

Strategy tiering (see ``plan.md`` § 3.11)
-----------------------------------------
* **Tier 1** (native LP — prompt is a feature/edge modifier): ``None``,
  ``GPF``, ``GPF-plus``, ``EdgePrompt``, ``EdgePromptplus``, ``UniPrompt``.
* **Tier 2** (LP via adapter — encoder wraps the GNN output through the
  prompt): ``Gprompt``, ``SelfPro``, ``ProNoG``, ``PSP``, ``DAGPrompT``,
  ``GraphPrompter``.
* **All-in-one** uses a separate edge-induced-graph adapter: each candidate
  edge is converted into a graph-level sample and then trained with HeavyPrompt.
* **Tier 3** (NOT yet adapted): ``GPPT`` (TaskToken weight-init is
  class-label-driven), ``MultiGprompt`` (custom Preprompt embed path),
  ``Prodigy`` (FiLM/support-set classifier), ``RELIEF`` (RL reward =
  classification accuracy). Constructor raises a clear capability error for
  these so callers fail fast.
"""

from __future__ import annotations

import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from prompt_graph.data import (
    load4link_prediction_multi_graph,
    load4link_prediction_single_graph,
)
from prompt_graph.utils import get_logger, sample_dir

from ..defines import GRAPH_TASKS, NODE_TASKS
from . import strategies as _strategies  # noqa: F401 -- registers bundled strategies
from .strategy import TaskContext  # noqa: F401 -- re-exported for ctx construction
from .task import BaseTask

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# Output dim for LinkTask is binary (positive vs negative edge).
LINK_OUTPUT_DIM = 2


# Strategies that are not yet adapted for LinkTask (see plan.md § 3.11).
LINK_TASK_UNSUPPORTED_PROMPTS = frozenset(
    {
        "GPPT",
        "MultiGprompt",
        "Prodigy",
        "RELIEF",
    }
)


# Strategies that LinkTask explicitly supports. All-in-one delegates to
# AllInOneLinkAdapter; the rest route through encode_nodes_for_link.
LINK_TASK_SUPPORTED_PROMPTS = frozenset(
    {
        "None",
        "All-in-one",
        "GPF",
        "GPF-plus",
        "EdgePrompt",
        "EdgePromptplus",
        "UniPrompt",
        "Gprompt",
        "SelfPro",
        "ProNoG",
        "PSP",
        "DAGPrompT",
        "GraphPrompter",
    }
)


# =============================================================================
# Data plumbing — splits + negative sampling
# =============================================================================


def _resolve_loader(dataset_name: str):
    """Return (data, input_dim) using the existing link-prediction loaders.

    The legacy ``load4link_prediction_{single,multi}_graph`` functions also
    return a concatenated (positive | negative) edge tensor + edge_label;
    LinkTask doesn't use those because it samples its own negatives per
    epoch with the full-positive exclusion mask.
    """
    if dataset_name in NODE_TASKS:
        data, _edge_label, _edge_index, input_dim, _output_dim = load4link_prediction_single_graph(
            dataset_name
        )
    elif dataset_name in GRAPH_TASKS:
        data, _edge_label, _edge_index, input_dim, _output_dim = load4link_prediction_multi_graph(
            dataset_name
        )
    else:
        raise ValueError(f"LinkTask does not support dataset: {dataset_name!r}")
    return data, input_dim


def _build_full_split(data, device, num_val: float = 0.05, num_test: float = 0.10):
    """RandomLinkSplit-based full split.

    The message-passing graph (``train_data.edge_index``) excludes val + test
    positives. ``edge_label_index`` / ``edge_label`` carry the supervision
    edges (positives only on train because ``add_negative_train_samples=False``).
    """
    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=False,
    )
    train_data, val_data, test_data = transform(data)
    return train_data.to(device), val_data.to(device), test_data.to(device)


def _subsample_train_positives(train_data, k_pos: int):
    """Restrict train_data's supervision edges to ``k_pos`` positives.

    Keeps ``train_data.edge_index`` (the message-passing graph) intact —
    transductive few-shot LP, matching what NodeTask/GraphTask do for their
    few-shot label subset.
    """
    pos_mask = train_data.edge_label == 1
    pos_pos = pos_mask.nonzero(as_tuple=False).view(-1)
    if pos_pos.numel() == 0 or k_pos <= 0:
        return train_data
    k = min(int(k_pos), pos_pos.numel())
    perm = torch.randperm(pos_pos.numel(), device=pos_pos.device)
    selected = pos_pos[perm[:k]]
    train_data.edge_label_index = train_data.edge_label_index[:, selected]
    train_data.edge_label = train_data.edge_label[selected]
    return train_data


def _build_full_positive_exclusion(train_data, val_data, test_data, device):
    """Union of train+val+test positive edges, used as ``negative_sampling`` exclusion.

    Without this, per-epoch negative sampling would happily emit val/test
    positives as training negatives, biasing both training and metrics.
    """
    pieces = []
    for d in (train_data, val_data, test_data):
        if d is None:
            continue
        if hasattr(d, "edge_label_index") and hasattr(d, "edge_label"):
            mask = d.edge_label == 1
            if mask.any():
                pieces.append(d.edge_label_index[:, mask])
        # train_data.edge_index is *also* a positive set (the message-passing
        # graph). For val/test it equals train_data.edge_index (RandomLinkSplit
        # contract), so we add train_data.edge_index once.
    pieces.append(train_data.edge_index)
    full_pos = torch.cat(pieces, dim=1).to(device)
    return full_pos


def _sample_train_negatives(num_nodes: int, exclude_edge_index, num_neg: int, device):
    """Sample ``num_neg`` negative edges avoiding all positives."""
    neg = negative_sampling(
        edge_index=exclude_edge_index,
        num_nodes=int(num_nodes),
        num_neg_samples=int(num_neg),
        method="sparse",
    )
    return neg.to(device)


def _sample_per_graph_negatives(batch_assignment, exclude_edge_index, num_neg_per_graph, device):
    """Sample negatives confined to each graph in a batched mega-graph.

    Avoids the cross-graph trivial-negative problem flagged by the
    rubber-duck review: a plain ``negative_sampling`` over a
    ``Batch.from_data_list([g1, g2, ...])`` will happily emit pairs ``(i, j)``
    with ``i`` in graph A and ``j`` in graph B — those are trivially negative
    and inflate AUROC. Here we sample negatives per-graph and concatenate.

    ``batch_assignment``: ``[N]`` long tensor mapping each node to its graph id
    (i.e. ``data.batch``).
    """
    if batch_assignment is None:
        raise ValueError("_sample_per_graph_negatives requires a `batch` assignment tensor.")

    batch_assignment = batch_assignment.to(device)
    exclude_edge_index = exclude_edge_index.to(device)
    src_g = batch_assignment[exclude_edge_index[0]]
    dst_g = batch_assignment[exclude_edge_index[1]]
    intra_mask = src_g == dst_g

    pieces = []
    num_graphs = int(batch_assignment.max().item()) + 1
    for g in range(num_graphs):
        node_mask = batch_assignment == g
        node_ids = node_mask.nonzero(as_tuple=False).view(-1)
        if node_ids.numel() < 2:
            continue

        # Local positives within this graph (re-indexed to local [0..n_g-1]).
        keep = intra_mask & (src_g == g)
        if keep.any():
            local_pos = exclude_edge_index[:, keep]
            # Map global -> local node ids.
            global_to_local = torch.full(
                (int(batch_assignment.size(0)),), -1, dtype=torch.long, device=device
            )
            global_to_local[node_ids] = torch.arange(node_ids.numel(), device=device)
            local_pos = global_to_local[local_pos]
        else:
            local_pos = torch.empty((2, 0), dtype=torch.long, device=device)

        local_neg = negative_sampling(
            edge_index=local_pos,
            num_nodes=int(node_ids.numel()),
            num_neg_samples=int(num_neg_per_graph),
            method="sparse",
        )
        if local_neg.numel() == 0:
            continue
        # Map local -> global.
        pieces.append(node_ids[local_neg])

    if not pieces:
        # Fallback to the global sampler to avoid an empty negative set;
        # caller will at least get a stable loss tensor.
        return _sample_train_negatives(
            int(batch_assignment.size(0)),
            exclude_edge_index,
            int(num_neg_per_graph) * max(1, num_graphs),
            device,
        )
    return torch.cat(pieces, dim=1).to(device)


# =============================================================================
# Prompt-aware node encoder dispatch
# =============================================================================


def encode_nodes_for_link(ctx: TaskContext, data, prompt_type: str) -> torch.Tensor:
    """Return ``[num_nodes, hid_dim]`` node embeddings adapted to ``prompt_type``.

    Strategies in :data:`LINK_TASK_UNSUPPORTED_PROMPTS` are guarded at
    LinkTask construction time and never reach this function.
    """
    x = data.x
    edge_index = data.edge_index

    if prompt_type == "None":
        return ctx.gnn(x, edge_index)

    if prompt_type in ("GPF", "GPF-plus"):
        x = ctx.prompt.add(x)
        return ctx.gnn(x, edge_index)

    if prompt_type in ("EdgePrompt", "EdgePromptplus"):
        return ctx.gnn(x, edge_index, prompt=ctx.prompt, prompt_type=prompt_type)

    if prompt_type == "Gprompt":
        z = ctx.gnn(x, edge_index)
        return ctx.prompt(z)

    if prompt_type == "SelfPro":
        # For LP we encode with the actual message-passing graph (Self-Pro's
        # identity-graph trick is classification-specific). The projector head
        # remains trainable; GNN is frozen by SelfProStrategy's setup.
        with torch.no_grad():
            embeds = ctx.gnn(x, edge_index)
        return F.normalize(ctx.prompt(embeds), p=2, dim=1)

    if prompt_type == "ProNoG":
        # ProNoG's full forward (neighborhood prompt + class prototypes) is
        # classification-specific. For LP we use just the element-wise
        # `selfprompt` weight applied to GNN embeddings — the part of ProNoG
        # that is task-agnostic. The neighborhood metanet + prototype path is
        # bypassed (a proper LP adapter would need an edge-aware prototype).
        embeds = ctx.gnn(x, edge_index)
        if hasattr(ctx.prompt, "selfprompt"):
            return ctx.prompt.selfprompt(embeds)
        return embeds

    if prompt_type == "PSP":
        # PSP's forward (feature_proj + label prototypes) is classification-
        # specific (feature_proj maps input_dim → hid_dim, not hid_dim →
        # hid_dim). For LP we fall back to bare GNN embeddings; PSP's
        # parameters get no gradient. A proper LP adapter would need a
        # node-level edge-prototype structure rather than label prototypes.
        return ctx.gnn(x, edge_index)

    if prompt_type == "DAGPrompT":
        h_list = ctx.gnn.forward_multihop(x, edge_index)
        h_list = ctx.prompt(h_list)
        return h_list[-1]

    if prompt_type == "GraphPrompter":
        # GraphPrompter's internal metagraph/decode is graph-level; for LP
        # we use the bare GNN embedding. Prompt parameters remain in the
        # optimizer for future adapter work.
        return ctx.gnn(x, edge_index)

    if prompt_type == "UniPrompt":
        from prompt_graph.tasker.strategies.uni_prompt import _fuse_and_embed

        return _fuse_and_embed(ctx, data, ctx.prompt, ctx.extra.get("tau", 0.99))

    # Fallback for any prompt that registered but doesn't have a LinkTask
    # encoder yet. Logged once so the gap is visible.
    logger.warning(
        "encode_nodes_for_link: no LinkTask encoder for prompt_type=%r; "
        "falling back to bare GNN. Add a dispatch branch in link_task.py.",
        prompt_type,
    )
    return ctx.gnn(x, edge_index)


# =============================================================================
# LinkTask class
# =============================================================================


class LinkTask(BaseTask):
    """Link-prediction downstream task.

    Parameters
    ----------
    data
        Either a single :class:`torch_geometric.data.Data` (single-graph LP,
        NODE_TASKS) or a batched mega-graph (multi-graph LP, GRAPH_TASKS).
    input_dim, output_dim
        ``input_dim`` is the feature dim; ``output_dim`` is forced to
        :data:`LINK_OUTPUT_DIM` (binary). The argument is accepted for
        API parity with NodeTask/GraphTask but ignored.
    task_num
        Number of folds to aggregate over (default 5, matches Node/GraphTask).
    num_neg_per_pos
        How many negative edges to sample per positive supervision edge each
        epoch. ``1`` matches the standard link-prediction protocol.
    """

    def __init__(
        self,
        data,
        input_dim,
        output_dim,  # noqa: ARG002 — forced to LINK_OUTPUT_DIM, kept for API parity
        task_num: int = 5,
        graphs_list=None,  # noqa: ARG002 — accepted for API parity, unused for LP
        num_neg_per_pos: int = 1,
        *args,
        **kwargs,
    ):
        self.aio_num_hops = kwargs.pop("aio_num_hops", 2)
        self.aio_max_nodes = kwargs.pop("aio_max_nodes", 64)
        self.aio_max_train_edges = kwargs.pop("aio_max_train_edges", None)
        super().__init__(*args, **kwargs)
        self.task_type = "LinkTask"
        self.task_num = int(task_num)
        self.num_neg_per_pos = int(num_neg_per_pos)

        if self.prompt_type in LINK_TASK_UNSUPPORTED_PROMPTS:
            raise NotImplementedError(
                f"LinkTask does not yet support prompt_type={self.prompt_type!r}. "
                f"See prompt_graph/tasker/link_task.py docstring (Tier 3) and "
                f"plan.md § 3.11 for the tracked adapter work. "
                f"Supported prompts: {sorted(LINK_TASK_SUPPORTED_PROMPTS)}."
            )

        self.data = data.to(self.device)
        self.input_dim = int(input_dim)
        self.output_dim = LINK_OUTPUT_DIM
        # Whether the underlying data is a batched mega-graph (multi-graph LP).
        # If so, negative sampling must respect per-graph membership.
        self.is_multi_graph = hasattr(self.data, "batch") and self.data.batch is not None

        # Make few-shot sample folders if shot_num > 0. The folders mirror
        # Node/Graph sample_data layout but carry per-fold seeds rather than
        # train/test idx tensors (LinkTask resamples splits per fold).
        if self.shot_num > 0:
            self._create_few_shot_folder()

    # ---- few-shot bookkeeping -----------------------------------------------

    def _create_few_shot_folder(self) -> None:
        k_shot_folder = str(sample_dir("Link", self.shot_num, self.dataset_name))
        os.makedirs(k_shot_folder, exist_ok=True)
        for i in range(1, self.task_num + 1):
            folder = os.path.join(k_shot_folder, str(i))
            os.makedirs(folder, exist_ok=True)
            seed_file = os.path.join(folder, "fold_seed.pt")
            if not os.path.exists(seed_file):
                # Deterministic per-fold seed derived from (dataset, shot, fold)
                # so reruns under the same args produce the same split.
                fold_seed = abs(hash((self.dataset_name, self.shot_num, i))) % (2**31)
                torch.save(torch.tensor(int(fold_seed)), seed_file)

    def _fold_seed(self, fold: int) -> int:
        seed_file = (
            sample_dir("Link", self.shot_num, self.dataset_name) / str(fold) / "fold_seed.pt"
        )
        if seed_file.exists():
            return int(torch.load(str(seed_file)).item())
        return abs(hash((self.dataset_name, self.shot_num, fold))) % (2**31)

    # ---- per-prompt TaskContext factory -------------------------------------

    def _ctx(self) -> TaskContext:
        """Build a TaskContext sufficient for any supported LinkTask strategy."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            answering=None,
            criterion=None,
            optimizer=self.optimizer,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.hid_dim,  # encoder output dim (for downstream that look at it)
            data=self.data,
            dataset_name=self.dataset_name,
            extra={
                "task_type": "LinkTask",
                "input_dim": self.input_dim,
                "lr": self.lr,
                "wd": self.wd,
                "prompt_type": self.prompt_type,
                "tau": getattr(self, "tau", 0.99),
            },
        )

    # ---- optimizer for LinkTask --------------------------------------------

    def _build_optimizer(self) -> None:
        """Adam over (GNN + prompt) parameters.

        LinkTask has no answering head — dot-product decoding is parameter-free.
        SelfPro's setup typically freezes the GNN; we honor that by only
        adding parameters that require grad.
        """
        params = [p for p in self.gnn.parameters() if p.requires_grad]
        if self.prompt is not None:
            params.extend(p for p in self.prompt.parameters() if p.requires_grad)
        if not params:
            raise RuntimeError("LinkTask optimizer has no trainable parameters.")
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

    # ---- BCE link loss + scorer --------------------------------------------

    @staticmethod
    def _link_scores(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Dot-product link scorer; returns ``[E]`` logits."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # ---- one training epoch -------------------------------------------------

    def _train_epoch(self, ctx: TaskContext, train_data, full_positives) -> float:
        self.gnn.train()
        if self.prompt is not None:
            self.prompt.train()
        self.optimizer.zero_grad()

        z = encode_nodes_for_link(ctx, train_data, self.prompt_type)

        pos_edge_index = train_data.edge_label_index
        num_pos = pos_edge_index.size(1)
        if self.is_multi_graph:
            # Per-graph negative sampling to avoid cross-graph trivial negatives.
            neg_edge_index = _sample_per_graph_negatives(
                getattr(train_data, "batch", None),
                full_positives,
                max(
                    1,
                    num_pos
                    * self.num_neg_per_pos
                    // max(1, int(train_data.batch.max().item()) + 1),
                ),
                self.device,
            )
        else:
            num_neg = max(1, num_pos * self.num_neg_per_pos)
            neg_edge_index = _sample_train_negatives(
                int(train_data.num_nodes), full_positives, num_neg, self.device
            )

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(num_pos, device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device),
            ]
        )

        logits = self._link_scores(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    # ---- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, ctx: TaskContext, eval_data, full_positives=None) -> tuple:
        """Returns (acc, f1, roc, prc) on the eval split's pos+neg labels.

        ``eval_data`` carries both positive and negative supervision edges
        (``RandomLinkSplit`` populates negatives on val/test even with
        ``add_negative_train_samples=False``).

        For multi-graph LP, RandomLinkSplit's eval negatives can include
        cross-graph pairs (trivially negative). When ``full_positives`` is
        provided AND ``self.is_multi_graph`` is True, we resample negatives
        per-graph against the full-positive exclusion set.
        """
        self.gnn.eval()
        if self.prompt is not None:
            self.prompt.eval()

        z = encode_nodes_for_link(ctx, eval_data, self.prompt_type)

        pos_mask = eval_data.edge_label == 1
        pos_edge_index = eval_data.edge_label_index[:, pos_mask]
        num_pos = int(pos_edge_index.size(1))

        if self.is_multi_graph and full_positives is not None and num_pos > 0:
            num_graphs = int(eval_data.batch.max().item()) + 1 if hasattr(eval_data, "batch") else 1
            neg_edge_index = _sample_per_graph_negatives(
                getattr(eval_data, "batch", None),
                full_positives,
                max(1, num_pos // max(1, num_graphs)),
                self.device,
            )
        else:
            neg_mask = eval_data.edge_label == 0
            neg_edge_index = eval_data.edge_label_index[:, neg_mask]

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat(
            [
                torch.ones(pos_edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device),
            ]
        ).long()

        logits = self._link_scores(z, edge_label_index)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        acc_m = torchmetrics.classification.BinaryAccuracy().to(self.device)
        f1_m = torchmetrics.classification.BinaryF1Score().to(self.device)
        auroc_m = torchmetrics.classification.BinaryAUROC().to(self.device)
        auprc_m = torchmetrics.classification.BinaryAveragePrecision().to(self.device)

        acc = acc_m(preds, labels)
        f1 = f1_m(preds, labels)
        roc = auroc_m(probs, labels)
        prc = auprc_m(probs, labels)
        return float(acc.item()), float(f1.item()), float(roc.item()), float(prc.item())

    # ---- per-fold run -------------------------------------------------------

    def _run_fold(self, fold: int) -> tuple:
        # Per-fold deterministic split: re-seed before RandomLinkSplit so the
        # fold's positives are reproducible. Cloning the source ``self.data``
        # keeps each fold's split independent.
        if self.shot_num > 0:
            torch.manual_seed(self._fold_seed(fold))

        fold_data = self.data.clone()
        train_data, val_data, test_data = _build_full_split(fold_data, self.device)
        if self.shot_num > 0:
            train_data = _subsample_train_positives(train_data, self.shot_num)

        full_positives = _build_full_positive_exclusion(
            train_data, val_data, test_data, self.device
        )

        # Re-initialise model + prompt + optimizer per fold (same convention
        # as NodeTask/GraphTask, so each fold is a fresh hyperparameter
        # evaluation rather than continuing from the previous fold's weights).
        self.initialize_gnn()
        self.initialize_prompt()
        self._build_optimizer()

        ctx = self._ctx()

        patience = 20
        best_val = -math.inf
        best_test = (0.0, 0.0, 0.0, 0.0)
        best_loss = float("inf")
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            loss = self._train_epoch(ctx, train_data, full_positives)
            if math.isnan(loss):
                logger.warning("Fold %d epoch %d: loss is NaN, stopping fold.", fold, epoch)
                break

            val_acc, val_f1, val_roc, val_prc = self._evaluate(ctx, val_data, full_positives)
            if val_roc > best_val:
                best_val = val_roc
                best_test = self._evaluate(ctx, test_data, full_positives)
                best_loss = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            logger.info(
                "Fold %d | Epoch %03d | Time %.3fs | Loss %.4f | Val AUROC %.4f",
                fold,
                epoch,
                time.time() - t0,
                loss,
                val_roc,
            )
            if cnt_wait >= patience:
                logger.info("Early stopping at epoch %d (fold %d).", epoch, fold)
                break

        return (*best_test, best_loss)

    # ---- public run ---------------------------------------------------------

    def run(self):
        if self.prompt_type == "All-in-one":
            from .all_in_one_link_adapter import AllInOneLinkAdapter

            return AllInOneLinkAdapter(self).run()

        accs, f1s, rocs, prcs, losses = [], [], [], [], []
        for fold in range(1, self.task_num + 1):
            acc, f1, roc, prc, loss = self._run_fold(fold)
            accs.append(acc)
            f1s.append(f1)
            rocs.append(roc)
            prcs.append(prc)
            if not math.isnan(loss) and not math.isinf(loss):
                losses.append(loss)
            logger.info(
                "Fold %d done | Test Acc %.4f | F1 %.4f | AUROC %.4f | AUPRC %.4f",
                fold,
                acc,
                f1,
                roc,
                prc,
            )

        mean_acc = float(np.mean(accs)) if accs else 0.0
        std_acc = float(np.std(accs)) if accs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        std_f1 = float(np.std(f1s)) if f1s else 0.0
        mean_roc = float(np.mean(rocs)) if rocs else 0.0
        std_roc = float(np.std(rocs)) if rocs else 0.0
        mean_prc = float(np.mean(prcs)) if prcs else 0.0
        std_prc = float(np.std(prcs)) if prcs else 0.0
        mean_loss = float(np.mean(losses)) if losses else float("inf")

        print(f" Final best | test Accuracy {mean_acc:.4f}±{std_acc:.4f}(std)")
        print(f" Final best | test F1 {mean_f1:.4f}±{std_f1:.4f}(std)")
        print(f" Final best | AUROC {mean_roc:.4f}±{std_roc:.4f}(std)")
        print(f" Final best | AUPRC {mean_prc:.4f}±{std_prc:.4f}(std)")
        logger.info(
            "%s %s %s Link Task completed", self.pre_train_type, self.gnn_type, self.prompt_type
        )

        return (
            mean_loss,
            mean_acc,
            std_acc,
            mean_f1,
            std_f1,
            mean_roc,
            std_roc,
            mean_prc,
            std_prc,
        )
