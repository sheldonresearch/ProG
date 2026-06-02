"""All-in-one adapter for LinkTask via edge-induced graph classification.

The All-in-one paper supports edge-level tasks by reformulating each candidate
edge into a graph-level sample: extract an induced subgraph around the two
endpoints, assign the edge label to that subgraph, then run HeavyPrompt +
graph-level answering. This adapter implements that path separately from
LinkTask's default dot-product decoder.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torchmetrics
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, subgraph

from prompt_graph.utils import get_logger

from .link_task import (
    _build_full_positive_exclusion,
    _build_full_split,
    _sample_per_graph_negatives,
    _sample_train_negatives,
    _subsample_train_positives,
)
from .strategy import TaskContext, get_strategy

logger = get_logger(__name__)


class AllInOneLinkAdapter:
    """Run All-in-one on LinkTask as edge-induced graph classification."""

    def __init__(self, task):
        self.task = task
        self.num_hops = int(getattr(task, "aio_num_hops", 2))
        self.max_nodes = int(getattr(task, "aio_max_nodes", 64))
        max_train_edges = getattr(task, "aio_max_train_edges", None)
        self.max_train_edges = None if max_train_edges is None else int(max_train_edges)

    def _trim_subset(self, subset: torch.Tensor, edge_pair: torch.Tensor) -> torch.Tensor:
        """Cap induced graph size while always retaining both target endpoints."""
        if self.max_nodes <= 0 or subset.numel() <= self.max_nodes:
            return subset

        target = torch.unique(edge_pair)
        keep_mask = ~torch.isin(subset, target)
        remaining = max(0, self.max_nodes - int(target.numel()))
        trimmed = torch.cat([target, subset[keep_mask][:remaining]])
        return torch.unique(trimmed)

    def _edge_induced_graph(self, data, edge_pair: torch.Tensor, label: int) -> Data:
        """Build one edge-induced graph and remove the queried edge if present."""
        edge_pair = edge_pair.to(data.edge_index.device).long()
        subset, _, _, _ = k_hop_subgraph(
            node_idx=edge_pair,
            num_hops=self.num_hops,
            edge_index=data.edge_index,
            relabel_nodes=False,
            num_nodes=int(data.num_nodes),
        )
        subset = self._trim_subset(subset, edge_pair)

        sub_edge_index, _ = subgraph(
            subset,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=int(data.num_nodes),
        )
        global_to_local = torch.full(
            (int(data.num_nodes),),
            -1,
            dtype=torch.long,
            device=data.edge_index.device,
        )
        global_to_local[subset] = torch.arange(subset.numel(), device=data.edge_index.device)
        local_pair = global_to_local[edge_pair]

        # Training positives are present in train_data.edge_index under
        # RandomLinkSplit. Remove the queried edge in both directions so the
        # graph classifier cannot solve the label by direct-edge leakage.
        if sub_edge_index.numel() > 0:
            forward = (sub_edge_index[0] == local_pair[0]) & (sub_edge_index[1] == local_pair[1])
            reverse = (sub_edge_index[0] == local_pair[1]) & (sub_edge_index[1] == local_pair[0])
            sub_edge_index = sub_edge_index[:, ~(forward | reverse)]

        return Data(
            x=data.x[subset],
            edge_index=sub_edge_index,
            y=torch.tensor(int(label), dtype=torch.long, device=data.edge_index.device),
        )

    def _cap_balanced_edges(
        self,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        max_edges: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_edges is None or edge_label.numel() <= max_edges:
            return edge_label_index, edge_label

        pos_idx = (edge_label == 1).nonzero(as_tuple=False).view(-1)
        neg_idx = (edge_label == 0).nonzero(as_tuple=False).view(-1)
        per_class = max(1, max_edges // 2)
        selected = torch.cat([pos_idx[:per_class], neg_idx[:per_class]])
        return edge_label_index[:, selected], edge_label[selected]

    def _graphs_from_edges(
        self,
        data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        *,
        max_edges: int | None = None,
    ) -> list[Data]:
        edge_label_index, edge_label = self._cap_balanced_edges(
            edge_label_index, edge_label.long(), max_edges
        )
        graphs = [
            self._edge_induced_graph(data, edge_label_index[:, i], int(edge_label[i].item()))
            for i in range(edge_label_index.size(1))
        ]
        if not graphs:
            raise ValueError("AllInOneLinkAdapter built an empty edge-induced graph dataset.")
        return graphs

    def _train_edges(self, train_data, full_positives) -> tuple[torch.Tensor, torch.Tensor]:
        pos_edge_index = train_data.edge_label_index
        num_pos = int(pos_edge_index.size(1))
        if self.task.is_multi_graph:
            num_graphs = int(train_data.batch.max().item()) + 1
            neg_edge_index = _sample_per_graph_negatives(
                getattr(train_data, "batch", None),
                full_positives,
                max(1, num_pos * self.task.num_neg_per_pos // max(1, num_graphs)),
                self.task.device,
            )
        else:
            neg_edge_index = _sample_train_negatives(
                int(train_data.num_nodes),
                full_positives,
                max(1, num_pos * self.task.num_neg_per_pos),
                self.task.device,
            )

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(pos_edge_index.size(1), device=self.task.device, dtype=torch.long),
                torch.zeros(neg_edge_index.size(1), device=self.task.device, dtype=torch.long),
            ]
        )
        return edge_label_index, edge_label

    def _eval_edges(self, eval_data, full_positives) -> tuple[torch.Tensor, torch.Tensor]:
        pos_mask = eval_data.edge_label == 1
        pos_edge_index = eval_data.edge_label_index[:, pos_mask]
        num_pos = int(pos_edge_index.size(1))

        if self.task.is_multi_graph and num_pos > 0:
            num_graphs = int(eval_data.batch.max().item()) + 1 if hasattr(eval_data, "batch") else 1
            neg_edge_index = _sample_per_graph_negatives(
                getattr(eval_data, "batch", None),
                full_positives,
                max(1, num_pos // max(1, num_graphs)),
                self.task.device,
            )
        else:
            neg_mask = eval_data.edge_label == 0
            neg_edge_index = eval_data.edge_label_index[:, neg_mask]

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(pos_edge_index.size(1), device=self.task.device, dtype=torch.long),
                torch.zeros(neg_edge_index.size(1), device=self.task.device, dtype=torch.long),
            ]
        )
        return edge_label_index, edge_label

    def _ctx(self, answer_epoch: int, prompt_epoch: int) -> TaskContext:
        return TaskContext(
            gnn=self.task.gnn,
            prompt=self.task.prompt,
            answering=self.task.answering,
            criterion=self.task.criterion,
            pg_opi=self.task.pg_opi,
            answer_opi=self.task.answer_opi,
            device=self.task.device,
            hid_dim=self.task.hid_dim,
            output_dim=self.task.output_dim,
            extra={
                "task_type": "LinkTask",
                "answer_epoch": answer_epoch,
                "prompt_epoch": prompt_epoch,
            },
        )

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float, float, float]:
        self.task.gnn.eval()
        self.task.prompt.eval()
        self.task.answering.eval()

        acc_m = torchmetrics.classification.BinaryAccuracy().to(self.task.device)
        f1_m = torchmetrics.classification.BinaryF1Score().to(self.task.device)
        auroc_m = torchmetrics.classification.BinaryAUROC().to(self.task.device)
        auprc_m = torchmetrics.classification.BinaryAveragePrecision().to(self.task.device)

        for batch in loader:
            batch = batch.to(self.task.device)
            prompted_graph = self.task.prompt(batch)
            graph_emb = self.task.gnn(
                prompted_graph.x,
                prompted_graph.edge_index,
                prompted_graph.batch,
            )
            probs = self.task.answering(graph_emb)
            positive_prob = probs[:, 1]
            labels = batch.y.long()
            preds = probs.argmax(dim=1).long()

            acc_m(preds, labels)
            f1_m(preds, labels)
            auroc_m(positive_prob, labels)
            auprc_m(positive_prob, labels)

        return (
            float(acc_m.compute().item()),
            float(f1_m.compute().item()),
            float(auroc_m.compute().item()),
            float(auprc_m.compute().item()),
        )

    def _run_fold(self, fold: int) -> tuple[float, float, float, float, float]:
        if self.task.shot_num > 0:
            torch.manual_seed(self.task._fold_seed(fold))

        train_data, val_data, test_data = _build_full_split(
            self.task.data.clone(), self.task.device
        )
        if self.task.shot_num > 0:
            train_data = _subsample_train_positives(train_data, self.task.shot_num)

        full_positives = _build_full_positive_exclusion(
            train_data, val_data, test_data, self.task.device
        )
        train_edge_index, train_edge_label = self._train_edges(train_data, full_positives)
        val_edge_index, val_edge_label = self._eval_edges(val_data, full_positives)
        test_edge_index, test_edge_label = self._eval_edges(test_data, full_positives)

        train_graphs = self._graphs_from_edges(
            train_data,
            train_edge_index,
            train_edge_label,
            max_edges=self.max_train_edges,
        )
        val_graphs = self._graphs_from_edges(val_data, val_edge_index, val_edge_label)
        test_graphs = self._graphs_from_edges(test_data, test_edge_index, test_edge_label)

        batch_size = max(1, int(self.task.batch_size))
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        self.task.initialize_gnn()
        self.task.answering = torch.nn.Sequential(
            torch.nn.Linear(self.task.hid_dim, self.task.output_dim),
            torch.nn.Softmax(dim=1),
        ).to(self.task.device)
        self.task.initialize_prompt()
        self.task.initialize_optimizer()

        answer_epoch = int(getattr(self.task, "answer_epoch", 1))
        prompt_epoch = int(getattr(self.task, "prompt_epoch", 1))
        strategy = get_strategy("All-in-one")()
        best_val = -math.inf
        best_test = (0.0, 0.0, 0.0, 0.0)
        best_loss = float("inf")
        cnt_wait = 0
        patience = 20

        for epoch in range(1, self.task.epochs + 1):
            t0 = time.time()
            loss = strategy.train_epoch(self._ctx(answer_epoch, prompt_epoch), train_loader)
            if math.isnan(loss):
                logger.warning("All-in-one LinkTask fold %d epoch %d: NaN loss.", fold, epoch)
                break

            _, _, val_roc, _ = self._evaluate(val_loader)
            if val_roc > best_val:
                best_val = val_roc
                best_test = self._evaluate(test_loader)
                best_loss = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            logger.info(
                "All-in-one Link fold %d | Epoch %03d | Time %.3fs | Loss %.4f | Val AUROC %.4f",
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

    def run(self):
        accs, f1s, rocs, prcs, losses = [], [], [], [], []
        for fold in range(1, self.task.task_num + 1):
            acc, f1, roc, prc, loss = self._run_fold(fold)
            accs.append(acc)
            f1s.append(f1)
            rocs.append(roc)
            prcs.append(prc)
            if not math.isnan(loss) and not math.isinf(loss):
                losses.append(loss)
            logger.info(
                "All-in-one Link fold %d done | Test Acc %.4f | F1 %.4f | AUROC %.4f | AUPRC %.4f",
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
            "%s %s All-in-one Link Task completed",
            self.task.pre_train_type,
            self.task.gnn_type,
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
