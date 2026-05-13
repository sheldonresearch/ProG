import os
import time
import warnings

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from prompt_graph.data import (
    GraphDataset,
    load4node,
    node_sample_and_save,
)
from prompt_graph.utils import (
    Gprompt_tuning_loss,
    center_embedding,
    constraint,
    get_logger,
    process,
    sample_dir,
)

from . import strategies as _strategies  # noqa: F401 -- registers bundled strategies
from .strategy import TaskContext, get_strategy
from .task import BaseTask

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class NodeTask(BaseTask):
    def __init__(self, data, input_dim, output_dim, task_num=5, graphs_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "NodeTask"
        self.task_num = task_num  # 增加task_num的参数，控制重复数量，默认为5
        self.data = data
        if self.dataset_name == "ogbn-arxiv":
            self.data.y = self.data.y.squeeze()
        self.graphs_list = graphs_list

        if self.prompt_type == "MultiGprompt":
            self.load_multigprompt_data()
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim

        self.create_few_data_folder()

    def create_few_data_folder(self):
        # 创建文件夹并保存数据
        k = self.shot_num  # shot_num 可变
        task_num = self.task_num  # task_num 可变
        for k in range(1, task_num + 1):
            k_shot_folder = str(sample_dir("Node", k, self.dataset_name))
            os.makedirs(k_shot_folder, exist_ok=True)

            for i in range(1, task_num + 1):
                folder = os.path.join(k_shot_folder, str(i))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    node_sample_and_save(self.data, k, folder, self.output_dim)
                    logger.info(str(k) + " shot " + str(i) + " th is saved!!")

    def load_multigprompt_data(self):
        adj, features, labels = process.load_data(self.dataset_name)
        # adj, features, labels = process.load_data(self.dataset_name)
        self.input_dim = features.shape[1]
        self.output_dim = labels.shape[1]
        logger.debug(f"a {self.output_dim}")
        features, _ = process.preprocess_features(features)
        self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
        self.labels = torch.FloatTensor(labels[np.newaxis])
        self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
        # print("labels",labels)
        logger.debug(f"adj {self.sp_adj.shape}")
        logger.debug(f"feature {features.shape}")

    def load_data(self):
        self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

    def train(self, data, train_idx):
        self.gnn.train()
        self.answering.train()
        self.optimizer.zero_grad()
        out = self.gnn(data.x, data.edge_index, batch=None)
        out = self.answering(out)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def GPPTtrain(self, data, train_idx):
        self.prompt.train()
        node_embedding = self.gnn(data.x, data.edge_index)
        out = self.prompt(node_embedding, data.edge_index)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
        self.pg_opi.zero_grad()
        loss.backward()
        self.pg_opi.step()
        mid_h = self.prompt.get_mid_h()
        self.prompt.update_StructureToken_weight(mid_h)
        return loss.item()

    def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
        self.DownPrompt.train()
        self.optimizer.zero_grad()
        prompt_feature = self.feature_prompt(self.features)
        # prompt_feature = self.feature_prompt(self.data.x)
        # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
        embeds1 = self.Preprompt.gcn(prompt_feature, self.sp_adj, True, False)
        pretrain_embs1 = embeds1[0, train_idx]
        logits = (
            self.DownPrompt(pretrain_embs, pretrain_embs1, train_lbls, 1).float().to(self.device)
        )
        loss = self.criterion(logits, train_lbls)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def SUPTtrain(self, data):
        self.gnn.train()
        self.optimizer.zero_grad()
        data.x = self.prompt.add(data.x)
        out = self.gnn(data.x, data.edge_index, batch=None)
        out = self.answering(out)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        orth_loss = self.prompt.orthogonal_loss()
        loss += orth_loss
        loss.backward()
        self.optimizer.step()
        return loss

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

    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        # we update answering and prompt alternately.
        # tune task head
        self.answering.train()
        self.prompt.eval()
        self.gnn.eval()
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
                f"frozen gnn | *tune prompt |frozen answering function... {epoch}/{prompt_epoch} ,loss: {pg_loss:.4f} "
            )

        # return pg_loss
        return answer_loss

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None
        for batch in train_loader:
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

    def _none_ctx(self):
        """Build a TaskContext for the 'None' strategy on this NodeTask."""
        return TaskContext(
            gnn=self.gnn,
            answering=self.answering,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={"task_type": "NodeTask"},
        )

    def _gpf_ctx(self):
        """Build a TaskContext for the GPF/GPF-plus strategy on this NodeTask."""
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
        """Build a TaskContext for the Gprompt strategy on this NodeTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            pg_opi=self.pg_opi,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
        )

    def _all_in_one_ctx(self):
        """Build a TaskContext for the All-in-one strategy on this NodeTask."""
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
                "task_type": "NodeTask",
                "answer_epoch": getattr(self, "answer_epoch", 1),
                "prompt_epoch": getattr(self, "prompt_epoch", 1),
            },
        )

    def _gppt_ctx(self):
        """Build a TaskContext for the GPPT strategy on this NodeTask."""
        return TaskContext(
            gnn=self.gnn,
            prompt=self.prompt,
            criterion=self.criterion,
            pg_opi=self.pg_opi,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={"task_type": "NodeTask"},
        )

    def _multi_gprompt_ctx(self, pretrain_embs, test_embs):
        """Build a TaskContext for the MultiGprompt strategy on this NodeTask.

        ``pretrain_embs`` / ``test_embs`` are precomputed once per fold in
        ``run`` because they depend on the fold-specific ``idx_train`` /
        ``idx_test`` indices.
        """
        return TaskContext(
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            extra={
                "task_type": "NodeTask",
                "Preprompt": self.Preprompt,
                "feature_prompt": self.feature_prompt,
                "DownPrompt": self.DownPrompt,
                "features": self.features,
                "sp_adj": self.sp_adj,
                "pretrain_embs": pretrain_embs,
                "test_embs": test_embs,
            },
        )

    def run(self):
        test_accs = []
        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.prompt_type == "All-in-one":
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = max(1, int(self.epochs / self.answer_epoch))
        for i in range(1, self.task_num + 1):
            sample_data_foler_path = (
                f"./Experiment/sample_data/Node/{self.dataset_name}/{self.shot_num}_shot/{i}"
            )

            if not os.path.exists(sample_data_foler_path):
                logger.warning(
                    f"Failed to find sample_data for shot {self.shot_num}, id {i}, path: {sample_data_foler_path}, skipping..."
                )
                continue

            self.initialize_gnn()
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(self.hid_dim, self.output_dim), torch.nn.Softmax(dim=1)
            ).to(self.device)
            self.initialize_prompt()
            self.initialize_optimizer()

            idx_train = (
                torch.load(f"{sample_data_foler_path}/train_idx.pt")
                .type(torch.long)
                .to(self.device)
            )
            logger.debug(f"idx_train {idx_train}")
            train_lbls = (
                torch.load(f"{sample_data_foler_path}/train_labels.pt")
                .type(torch.long)
                .squeeze()
                .to(self.device)
            )
            logger.debug(f"true {i} {train_lbls}")
            idx_test = (
                torch.load(f"{sample_data_foler_path}/test_idx.pt").type(torch.long).to(self.device)
            )
            test_lbls = (
                torch.load(f"{sample_data_foler_path}/test_labels.pt")
                .type(torch.long)
                .squeeze()
                .to(self.device)
            )

            # GPPT prompt initialtion
            if self.prompt_type == "GPPT":
                node_embedding = self.gnn(self.data.x, self.data.edge_index)
                self.prompt.weight_init(
                    node_embedding, self.data.edge_index, self.data.y, idx_train
                )

            if self.prompt_type in ["Gprompt", "All-in-one", "GPF", "GPF-plus"]:
                train_graphs = []
                test_graphs = []
                # self.graphs_list.to(self.device)
                logger.info("distinguishing the train dataset and test dataset...")
                for graph in self.graphs_list:
                    if graph.index in idx_train:
                        train_graphs.append(graph)
                    elif graph.index in idx_test:
                        test_graphs.append(graph)
                logger.info("Done!!!")

                train_dataset = GraphDataset(train_graphs)
                test_dataset = GraphDataset(test_graphs)

                # 创建数据加载器
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                logger.info("prepare induced graph data is finished!")

            if self.prompt_type == "MultiGprompt":
                embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                pretrain_embs = embeds[0, idx_train]
                test_embs = embeds[0, idx_test]

            patience = 20
            best = 1e9
            cnt_wait = 0

            # Stateful strategies (e.g. Gprompt's mean_centers) need a single
            # instance reused across this fold's train_epoch + evaluate calls.
            gprompt_strategy = get_strategy("Gprompt")() if self.prompt_type == "Gprompt" else None

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                if self.prompt_type == "None":
                    loss = get_strategy("None")().train_epoch(
                        self._none_ctx(), (self.data, idx_train)
                    )
                elif self.prompt_type == "GPPT":
                    loss = get_strategy("GPPT")().train_epoch(
                        self._gppt_ctx(), (self.data, idx_train)
                    )
                elif self.prompt_type == "All-in-one":
                    loss = get_strategy("All-in-one")().train_epoch(
                        self._all_in_one_ctx(), train_loader
                    )
                elif self.prompt_type in ["GPF", "GPF-plus"]:
                    loss = get_strategy(self.prompt_type)().train_epoch(
                        self._gpf_ctx(), train_loader
                    )
                elif self.prompt_type == "Gprompt":
                    loss = gprompt_strategy.train_epoch(self._gprompt_ctx(), train_loader)
                elif self.prompt_type == "MultiGprompt":
                    loss = get_strategy("MultiGprompt")().train_epoch(
                        self._multi_gprompt_ctx(pretrain_embs, test_embs),
                        (train_lbls, idx_train),
                    )

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

                logger.info(
                    f"Epoch {epoch:03d} |  Time(s) {time.time() - t0:.4f} | Loss {loss:.4f}  "
                )
            import math

            if not math.isnan(loss):
                batch_best_loss.append(loss)

                if self.prompt_type == "None":
                    test_acc, f1, roc, prc = get_strategy("None")().evaluate(
                        self._none_ctx(), (self.data, idx_test)
                    )
                elif self.prompt_type == "GPPT":
                    test_acc, f1, roc, prc = get_strategy("GPPT")().evaluate(
                        self._gppt_ctx(), (self.data, idx_test)
                    )
                elif self.prompt_type == "All-in-one":
                    test_acc, f1, roc, prc = get_strategy("All-in-one")().evaluate(
                        self._all_in_one_ctx(), test_loader
                    )
                elif self.prompt_type in ["GPF", "GPF-plus"]:
                    test_acc, f1, roc, prc = get_strategy(self.prompt_type)().evaluate(
                        self._gpf_ctx(), test_loader
                    )
                elif self.prompt_type == "Gprompt":
                    test_acc, f1, roc, prc = gprompt_strategy.evaluate(
                        self._gprompt_ctx(), test_loader
                    )
                elif self.prompt_type == "MultiGprompt":
                    test_acc, f1, roc, prc = get_strategy("MultiGprompt")().evaluate(
                        self._multi_gprompt_ctx(pretrain_embs, test_embs),
                        (test_lbls, idx_test),
                    )

                logger.info(
                    f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}"
                )
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
        logger.info(f"Acc List {test_accs}")  # 输出所有测试的Acc结果
        print(f" Final best | test Accuracy {mean_test_acc:.4f}±{std_test_acc:.4f}(std)")
        print(f" Final best | test F1 {mean_f1:.4f}±{std_f1:.4f}(std)")
        print(f" Final best | AUROC {mean_roc:.4f}±{std_roc:.4f}(std)")
        print(f" Final best | AUPRC {mean_prc:.4f}±{std_prc:.4f}(std)")

        logger.info(f"{self.pre_train_type} {self.gnn_type} {self.prompt_type} Node Task completed")
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
