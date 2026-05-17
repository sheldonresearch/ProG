import math
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from prompt_graph.data import graph_sample_and_save, load4graph
from prompt_graph.utils import (
    Gprompt_tuning_loss,
    center_embedding,
    constraint,
    get_logger,
    sample_dir,
)

from ..defines import GRAPH_TASKS
from . import strategies as _strategies  # noqa: F401 -- registers bundled strategies
from .strategy import TaskContext, get_strategy
from .task import BaseTask

logger = get_logger(__name__)


class GraphTask(BaseTask):
    def __init__(self, input_dim, output_dim, dataset, task_num=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "GraphTask"
        self.task_num = task_num
        # self.load_data()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        if self.shot_num > 0:
            self.create_few_data_folder()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering = torch.nn.Sequential(
            torch.nn.Linear(self.hid_dim, self.output_dim), torch.nn.Softmax(dim=1)
        ).to(self.device)
        self.initialize_optimizer()

    def create_few_data_folder(self):
        # 创建文件夹并保存数据
        k = self.shot_num
        k_shot_folder = str(sample_dir("Graph", k, self.dataset_name))
        os.makedirs(k_shot_folder, exist_ok=True)
        for i in range(1, self.task_num + 1):
            folder = os.path.join(k_shot_folder, str(i))
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
                graph_sample_and_save(self.dataset, k, folder, self.output_dim)
                logger.info(str(k) + " shot " + str(i) + " th is saved!!")

    def load_data(self):
        if self.dataset_name in GRAPH_TASKS:
            self.input_dim, self.output_dim, self.dataset = load4graph(
                self.dataset_name, self.shot_num
            )

    def node_degree_as_features(self, data_list):
        from torch_geometric.utils import degree

        for data in data_list:
            # 计算所有节点的度数，这将返回一个张量
            deg = degree(data.edge_index[0], dtype=torch.long)

            # 将度数张量变形为[nodes, 1]以便与其他特征拼接
            deg = deg.view(-1, 1).float()

            # 如果原始数据没有节点特征，可以直接使用度数作为特征
            if data.x is None:
                data.x = deg
            else:
                # 将度数特征拼接到现有的节点特征上
                data.x = torch.cat([data.x, deg], dim=1)

    def Train(self, train_loader):

        self.gnn.train()
        total_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        # we update answering and prompt alternately.

        # answer_epoch = 1  # 50
        # prompt_epoch = 1  # 50
        # answer_epoch = 5  # 50  #PROTEINS # COX2
        # prompt_epoch = 1  # 50

        # tune task head
        self.answering.train()
        self.prompt.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(
                train_loader, self.gnn, self.answering, self.criterion, self.answer_opi, self.device
            )
            logger.info(
                f"frozen gnn | frozen prompt | *tune answering function... {epoch}/{answer_epoch} ,loss: {answer_loss:.4f} "
            )

        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune(
                train_loader, self.gnn, self.answering, self.criterion, self.pg_opi, self.device
            )
            logger.info(
                f"frozen gnn | *tune prompt |frozen answering function... {epoch}/{answer_epoch} ,loss: {pg_loss:.4f} "
            )

        return pg_loss

    def GPFTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            batch.x = self.prompt.add(batch.x)
            out = self.gnn(
                batch.x,
                batch.edge_index,
                batch.batch,
                prompt=self.prompt,
                prompt_type=self.prompt_type,
            )
            out = self.answering(out)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None

        for batch in train_loader:
            # archived code for complete prototype embeddings of each labels. Not as well as batch version
            # # compute the prototype embeddings of each type of label

            self.pg_opi.zero_grad()
            batch = batch.to(self.device)
            out = self.gnn(
                batch.x, batch.edge_index, batch.batch, prompt=self.prompt, prompt_type="Gprompt"
            )
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out, batch.y, self.output_dim)
            # 累积中心向量和样本数
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)
            loss.backward()
            self.pg_opi.step()
            total_loss += loss.item()
        # 计算加权平均中心向量
        mean_centers = accumulated_centers / accumulated_counts

        return total_loss / len(train_loader), mean_centers

    def GPPTtrain(self, train_loader):
        self.prompt.train()
        for batch in train_loader:
            temp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            graph_list = batch.to_data_list()
            for index, graph in enumerate(graph_list):
                graph = graph.to(self.device)
                node_embedding = self.gnn(graph.x, graph.edge_index)
                out = self.prompt(
                    node_embedding, graph.edge_index
                )  # gppt下游在1-shot的时候，prompt结果为nan
                loss = self.criterion(
                    out,
                    torch.full((1, graph.x.shape[0]), graph.y.item()).reshape(-1).to(self.device),
                )
                temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            temp_loss = temp_loss / (index + 1)
            self.pg_opi.zero_grad()
            temp_loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
        return temp_loss.item()

    def _gppt_weight_init(self, train_dataset, test_dataset, processed_dataset, idx_train):
        """Run GPPT prompt.weight_init using the appropriate train_node_ids construction.

        - idx_train is not None → few-shot path: train_node_ids must be remapped
          to global indices via ``idx_train`` (the original Branch A behaviour).
        - idx_train is None → full-dataset path: train_node_ids are a simple range
          over train_dataset.x rows (the original Branch B behaviour).

        Returns a (possibly updated) test_loader matching the per-sample evaluation
        loader the original code used after weight_init.
        """
        if self.dataset_name in ["COLLAB", "IMDB-BINARY", "REDDIT-BINARY", "ogbg-ppa"]:
            total_num_nodes = sum([data.num_nodes for data in train_dataset])
            train_node_ids = torch.arange(0, total_num_nodes).squeeze().to(self.device)
            self.gppt_loader = DataLoader(
                processed_dataset.to_data_list(), batch_size=1, shuffle=False
            )
            for i, batch in enumerate(self.gppt_loader):
                if i == 0:
                    node_for_graph_labels = torch.full((1, batch.x.shape[0]), batch.y.item())
                    node_embedding = self.gnn(
                        batch.x.to(self.device), batch.edge_index.to(self.device)
                    )
                else:
                    node_for_graph_labels = torch.concat(
                        [node_for_graph_labels, torch.full((1, batch.x.shape[0]), batch.y.item())],
                        dim=1,
                    )
                    node_embedding = torch.concat(
                        [
                            node_embedding,
                            self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device)),
                        ],
                        dim=0,
                    )

            node_for_graph_labels = node_for_graph_labels.reshape(-1).to(self.device)
            self.prompt.weight_init(
                node_embedding,
                processed_dataset.edge_index.to(self.device),
                node_for_graph_labels,
                train_node_ids,
            )
        else:
            train_node_ids = torch.arange(0, train_dataset.x.shape[0]).squeeze().to(self.device)
            if idx_train is not None:
                # 将子图的节点id转换为全图的节点id (Branch A few-shot remap)
                iterate_id_num = 0
                for index, g in enumerate(train_dataset):
                    current_node_ids = iterate_id_num + torch.arange(0, g.x.shape[0]).squeeze().to(
                        self.device
                    )
                    iterate_id_num += g.x.shape[0]
                    previous_node_num = sum(
                        [self.dataset[i].x.shape[0] for i in range(idx_train[index] - 1)]
                    )
                    train_node_ids[current_node_ids] += previous_node_num

            self.gppt_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            for i, batch in enumerate(self.gppt_loader):
                if i == 0:
                    node_for_graph_labels = torch.full((1, batch.x.shape[0]), batch.y.item())
                else:
                    node_for_graph_labels = torch.concat(
                        [node_for_graph_labels, torch.full((1, batch.x.shape[0]), batch.y.item())],
                        dim=1,
                    )

            node_embedding = self.gnn(
                self.dataset.x.to(self.device), self.dataset.edge_index.to(self.device)
            )
            node_for_graph_labels = node_for_graph_labels.reshape(-1).to(self.device)
            self.prompt.weight_init(
                node_embedding,
                self.dataset.edge_index.to(self.device),
                node_for_graph_labels,
                train_node_ids,
            )

        return DataLoader(test_dataset, batch_size=1, shuffle=False)

    def _run_for_split(
        self,
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        processed_dataset=None,
        idx_train=None,
        answer_epoch=50,
        prompt_epoch=50,
    ):
        """Run the full epoch+train+eval+metrics loop on a single train/test split.

        This is the body shared by the few-shot (Branch A) and full-dataset
        (Branch B) paths of ``run``. Behavior parameters that historically
        differed between the two branches (``answer_epoch``/``prompt_epoch`` for
        All-in-one, and the GPPT weight-init train_node_ids construction via
        ``idx_train``) are passed in as arguments.

        Returns ``(test_acc, f1, roc, prc, loss)`` — caller decides how to
        aggregate. ``loss`` is the final epoch loss (used by Branch A for the
        per-fold best-loss list).
        """
        patience = 20
        best = 1e9
        cnt_wait = 0

        if self.prompt_type == "GPPT":
            test_loader = self._gppt_weight_init(
                train_dataset, test_dataset, processed_dataset, idx_train
            )

        # Stateful strategies (e.g. Gprompt's mean_centers) need a single
        # instance reused across this fold's train_epoch + evaluate calls.
        gprompt_strategy = get_strategy("Gprompt")() if self.prompt_type == "Gprompt" else None

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            if self.prompt_type == "None":
                loss = get_strategy("None")().train_epoch(self._none_ctx(), train_loader)
            elif self.prompt_type == "All-in-one":
                loss = get_strategy("All-in-one")().train_epoch(
                    self._all_in_one_ctx(answer_epoch, prompt_epoch),
                    train_loader,
                )
            elif self.prompt_type in ["GPF", "GPF-plus"]:
                loss = get_strategy(self.prompt_type)().train_epoch(self._gpf_ctx(), train_loader)
            elif self.prompt_type == "Gprompt":
                loss = gprompt_strategy.train_epoch(self._gprompt_ctx(), train_loader)
            elif self.prompt_type == "GPPT":
                loss = get_strategy("GPPT")().train_epoch(self._gppt_ctx(), train_loader)

            if loss < best:
                best = loss
                # best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), args.save_name)
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    logger.info("-" * 100)
                    logger.info("Early stopping at " + str(epoch) + " epoch!")
                    break
            logger.info(f"Epoch {epoch:03d} |  Time(s) {time.time() - t0:.4f} | Loss {loss:.4f}  ")

        logger.info("Begin to evaluate")

        if self.prompt_type == "None":
            test_acc, f1, roc, prc = get_strategy("None")().evaluate(self._none_ctx(), test_loader)
        elif self.prompt_type == "GPPT":
            test_acc, f1, roc, prc = get_strategy("GPPT")().evaluate(self._gppt_ctx(), test_loader)
        elif self.prompt_type == "All-in-one":
            test_acc, f1, roc, prc = get_strategy("All-in-one")().evaluate(
                self._all_in_one_ctx(answer_epoch, prompt_epoch),
                test_loader,
            )
        elif self.prompt_type in ["GPF", "GPF-plus"]:
            test_acc, f1, roc, prc = get_strategy(self.prompt_type)().evaluate(
                self._gpf_ctx(), test_loader
            )
        elif self.prompt_type == "Gprompt":
            test_acc, f1, roc, prc = gprompt_strategy.evaluate(self._gprompt_ctx(), test_loader)

        logger.info(
            f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}"
        )

        return test_acc, f1, roc, prc, loss

    def _none_ctx(self):
        """Build a TaskContext for the 'None' strategy on this GraphTask."""
        return TaskContext(
            gnn=self.gnn,
            answering=self.answering,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={"task_type": "GraphTask"},
        )

    def _gpf_ctx(self):
        """Build a TaskContext for the GPF/GPF-plus strategy on this GraphTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            answering=self.answering,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={"input_dim": self.input_dim},
        )

    def _gprompt_ctx(self):
        """Build a TaskContext for the Gprompt strategy on this GraphTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            pg_opi=self.pg_opi,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
        )

    def _all_in_one_ctx(self, answer_epoch=1, prompt_epoch=1):
        """Build a TaskContext for the All-in-one strategy on this GraphTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            answering=self.answering,
            criterion=self.criterion,
            pg_opi=self.pg_opi,
            answer_opi=self.answer_opi,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={
                "task_type": "GraphTask",
                "answer_epoch": answer_epoch,
                "prompt_epoch": prompt_epoch,
            },
        )

    def _gppt_ctx(self):
        """Build a TaskContext for the GPPT strategy on this GraphTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            criterion=self.criterion,
            pg_opi=self.pg_opi,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={"task_type": "GraphTask"},
        )

    def run(self):
        # Dispatch to ``_run_for_split``. Few-shot path loops over 5 sample_data
        # folds and returns aggregated stats (9-tuple); full-dataset path runs a
        # single split (4-tuple). The 9-tuple vs. 4-tuple return-shape mismatch
        # predates this refactor.
        if self.prompt_type == "All-in-one":
            # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
            # self.prompt_epoch = 1
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = max(1, int(self.epochs / self.answer_epoch))

        if self.shot_num > 0:
            test_accs = []
            f1s = []
            rocs = []
            prcs = []
            batch_best_loss = []
            for i in range(1, self.task_num + 1):
                split_dir = sample_dir("Graph", self.shot_num, self.dataset_name) / str(i)
                idx_train = (
                    torch.load(str(split_dir / "train_idx.pt"))
                    .type(torch.long)
                    .to(self.device)
                )
                logger.debug(f"idx_train {idx_train}")
                train_lbls = (
                    torch.load(str(split_dir / "train_labels.pt"))
                    .type(torch.long)
                    .squeeze()
                    .to(self.device)
                )
                logger.debug(f"true {i} {train_lbls}")

                idx_test = (
                    torch.load(str(split_dir / "test_idx.pt"))
                    .type(torch.long)
                    .to(self.device)
                )

                train_dataset = self.dataset[idx_train]
                test_dataset = self.dataset[idx_test]

                processed_dataset = None
                if self.dataset_name in ["COLLAB", "IMDB-BINARY", "REDDIT-BINARY", "ogbg-ppa"]:
                    from torch_geometric.data import Batch

                    train_dataset = [train_g for train_g in train_dataset]
                    test_dataset = [test_g for test_g in test_dataset]
                    self.node_degree_as_features(train_dataset)
                    self.node_degree_as_features(test_dataset)
                    if self.prompt_type == "GPPT":
                        processed_dataset = [g for g in self.dataset]
                        self.node_degree_as_features(processed_dataset)
                        processed_dataset = Batch.from_data_list([g for g in processed_dataset])
                    self.input_dim = train_dataset[0].x.size(1)

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                logger.info("prepare data is finished!")

                test_acc, f1, roc, prc, loss = self._run_for_split(
                    train_dataset,
                    test_dataset,
                    train_loader,
                    test_loader,
                    processed_dataset=processed_dataset,
                    idx_train=idx_train,
                    answer_epoch=getattr(self, "answer_epoch", 50),
                    prompt_epoch=getattr(self, "prompt_epoch", 50),
                )

                if not math.isnan(loss):
                    batch_best_loss.append(loss)
                logger.info(f"best_loss {batch_best_loss}")
                test_accs.append(test_acc)
                f1s.append(f1)
                rocs.append(roc)
                prcs.append(prc)

            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            mean_roc = np.mean(rocs)
            std_roc = np.std(rocs)
            mean_prc = np.mean(prcs)
            std_prc = np.std(prcs)
            print(f" Final best | test Accuracy {mean_test_acc:.4f}±{std_test_acc:.4f}(std)")
            print(f" Final best | test F1 {mean_f1:.4f}±{std_f1:.4f}(std)")
            print(f" Final best | AUROC {mean_roc:.4f}±{std_roc:.4f}(std)")
            print(f" Final best | AUPRC {mean_prc:.4f}±{std_prc:.4f}(std)")

            logger.info(
                f"{self.pre_train_type} {self.gnn_type} {self.prompt_type}  Graph Task completed"
            )
            mean_best = np.mean(batch_best_loss)

            return (
                mean_best,
                mean_test_acc,
                std_test_acc,
                mean_f1,
                std_f1,
                mean_roc,
                std_roc,
                mean_prc,
                std_prc,
            )

        else:
            train_dataset, test_dataset = self.dataset

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            logger.info("prepare data is finished!")

            if self.prompt_type == "All-in-one":
                # self.answer_epoch = 5 MUTAG Graph MAE / GraphCL
                # self.prompt_epoch = 1
                self.answer_epoch = 5
                self.prompt_epoch = 1
                self.epochs = max(1, int(self.epochs / self.answer_epoch))

            test_acc, f1, roc, prc, _ = self._run_for_split(
                train_dataset,
                test_dataset,
                train_loader,
                test_loader,
                processed_dataset=None,
                idx_train=None,
                answer_epoch=getattr(self, "answer_epoch", 50),
                prompt_epoch=getattr(self, "prompt_epoch", 50),
            )

            logger.info(
                f"{self.pre_train_type} {self.gnn_type} {self.prompt_type}  Graph Task completed"
            )

            return test_acc, f1, roc, prc
