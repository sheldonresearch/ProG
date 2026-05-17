import os

import torch
from torch import optim

from prompt_graph.model import build_gnn
from prompt_graph.pretrain import NodePrePrompt
from prompt_graph.prompt import (
    GPF,
    GPF_plus,
    GPPTPrompt,
    Gprompt,
    HeavyPrompt,
    downprompt,
    featureprompt,
)
from prompt_graph.utils import Gprompt_tuning_loss, get_logger, resolve_device

logger = get_logger(__name__)


class BaseTask:
    def __init__(
        self,
        pre_train_model_path="None",
        gnn_type="TransformerConv",
        hid_dim=128,
        num_layer=2,
        dataset_name="Cora",
        prompt_type="None",
        epochs=100,
        shot_num=10,
        device=5,
        lr=0.001,
        wd=5e-4,
        batch_size=16,
        search=False,
    ):

        self.pre_train_model_path = pre_train_model_path
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        self.device = self._resolve_device(device)
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.search = search
        self.initialize_lossfn()

    @staticmethod
    def _resolve_device(device):
        """统一的设备解析；委托给 prompt_graph.utils.resolve_device。"""
        return resolve_device(device)

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.prompt_type == "Gprompt":
            self.criterion = Gprompt_tuning_loss()

    def initialize_optimizer(self):
        if self.prompt_type == "None":
            if self.pre_train_model_path == "None":
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            else:
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                # self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)

        elif self.prompt_type == "All-in-one":
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=1e-6, weight_decay=self.wd)
            self.answer_opi = optim.Adam(
                self.answering.parameters(), lr=self.lr, weight_decay=self.wd
            )
        elif self.prompt_type in ["GPF", "GPF-plus"]:
            model_param_group = []
            model_param_group.append({"params": self.prompt.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ["Gprompt"]:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ["GPPT"]:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=2e-3, weight_decay=5e-4)
        elif self.prompt_type == "SelfPro":
            # Freeze GNN; only optimise the projector
            for p in self.gnn.parameters():
                p.requires_grad = False
            self.optimizer = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type == "ProNoG":
            # Freeze GNN; only optimise prompt parameters
            for p in self.gnn.parameters():
                p.requires_grad = False
            self.optimizer = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type == "DAGPrompT":
            params = [
                *self.prompt.parameters(),
                *self.param_center_embeddings.parameters(),
            ]
            self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ("PSP", "RELIEF", "GraphPrompter"):
            model_param_group = []
            model_param_group.append({"params": self.prompt.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type == "MultiGprompt":
            self.optimizer = optim.Adam(
                [*self.DownPrompt.parameters(), *self.feature_prompt.parameters()], lr=self.lr
            )

    def initialize_prompt(self):
        if self.prompt_type == "None":
            self.prompt = None
        elif self.prompt_type == "GPPT":
            if self.task_type == "NodeTask":
                if self.dataset_name == "Texas":
                    self.prompt = GPPTPrompt(self.hid_dim, 5, self.output_dim, device=self.device)
                else:
                    self.prompt = GPPTPrompt(
                        self.hid_dim, self.output_dim, self.output_dim, device=self.device
                    )
            elif self.task_type == "GraphTask":
                self.prompt = GPPTPrompt(
                    self.hid_dim, self.output_dim, self.output_dim, device=self.device
                )
        elif self.prompt_type == "All-in-one":
            # lr, wd = 0.001, 0.00001
            # self.prompt = LightPrompt(token_dim=self.input_dim, token_num_per_group=100, group_num=self.output_dim, inner_prune=0.01).to(self.device)
            if self.task_type == "NodeTask":
                self.prompt = HeavyPrompt(
                    token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3
                ).to(self.device)
            elif self.task_type == "GraphTask":
                self.prompt = HeavyPrompt(
                    token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3
                ).to(self.device)
        elif self.prompt_type == "GPF":
            self.prompt = GPF(self.input_dim).to(self.device)
        elif self.prompt_type == "GPF-plus":
            self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
        # elif self.prompt_type == 'sagpool':
        #     self.prompt = SAGPoolPrompt(self.input_dim , num_clusters=5, ratio=0.5).to(self.device)
        # elif self.prompt_type == 'diffpool':
        #     self.prompt = DiffPoolPrompt(self.input_dim, num_clusters=5 ).to(self.device)
        elif self.prompt_type == "Gprompt":
            self.prompt = Gprompt(self.hid_dim).to(self.device)
        elif self.prompt_type == "Prodigy":
            from prompt_graph.prompt.ProdigyPrompt import ProdigyPrompt
            self.prompt = ProdigyPrompt(self.hid_dim).to(self.device)
        elif self.prompt_type == "EdgePrompt":
            from prompt_graph.prompt.EdgePrompt import EdgePrompt
            dim_list = [self.hid_dim] * (self.num_layer - 1) + [self.hid_dim]
            self.prompt = EdgePrompt(dim_list).to(self.device)
        elif self.prompt_type == "EdgePromptplus":
            from prompt_graph.prompt.EdgePrompt import EdgePromptplus
            dim_list = [self.hid_dim] * (self.num_layer - 1) + [self.hid_dim]
            self.prompt = EdgePromptplus(dim_list, num_anchors=20).to(self.device)
        elif self.prompt_type == "UniPrompt":
            from prompt_graph.prompt.UniPrompt import UniPrompt
            self.prompt = UniPrompt(self.data.x, num_nodes=self.data.num_nodes).to(self.device)
        elif self.prompt_type == "SelfPro":
            from prompt_graph.prompt.SelfProPrompt import SelfProPrompt
            self.prompt = SelfProPrompt(self.hid_dim, self.hid_dim, num_layers=1).to(self.device)
        elif self.prompt_type == "ProNoG":
            from prompt_graph.prompt.ProNoGPrompt import ProNoGPrompt
            self.gnn.eval()
            with torch.no_grad():
                embeds = self.gnn(self.data.x, self.data.edge_index).detach()
            from prompt_graph.tasker.strategies.pro_no_g import _build_neighbor_lists
            neighbors, neighbors_2hop = _build_neighbor_lists(
                self.data.edge_index, self.data.num_nodes, self.device
            )
            self.prompt = ProNoGPrompt(
                embeds=embeds,
                neighbors=neighbors,
                neighbors_2hop=neighbors_2hop,
                nb_classes=self.output_dim,
                hidden_size=self.hid_dim,
            ).to(self.device)
        elif self.prompt_type == "DAGPrompT":
            from prompt_graph.prompt.DAGPrompT import (
                DAGPrompt,
                ParameterizedMultiHopCenterEmbedding,
            )
            hop_range = self.num_layer + 1
            dim_list = [self.input_dim] + [self.hid_dim] * self.num_layer
            self.prompt = DAGPrompt(dim_list).to(self.device)
            self.param_center_embeddings = ParameterizedMultiHopCenterEmbedding(
                hop_num=hop_range, label_num=self.output_dim, hidden_dim=self.hid_dim
            ).to(self.device)
        elif self.prompt_type == "PSP":
            from prompt_graph.prompt.PSPPrompt import PSPPrompt
            self.prompt = PSPPrompt(
                self.data.num_nodes, self.output_dim, self.input_dim, self.hid_dim
            ).to(self.device)
        elif self.prompt_type == "RELIEF":
            from prompt_graph.prompt.RELIEFPrompt import RELIEFPrompt
            self.prompt = RELIEFPrompt(self.data.num_nodes, self.input_dim).to(self.device)
        elif self.prompt_type == "GraphPrompter":
            from prompt_graph.prompt.GraphPrompterPrompt import GraphPrompterPrompt
            task_type_str = "graph" if self.task_type == "GraphTask" else "node"
            self.prompt = GraphPrompterPrompt(
                emb_dim=self.hid_dim,
                num_classes=self.output_dim,
                num_prompts=getattr(self, "gp_num_prompts", 10),
                shots=getattr(self, "shot_num", 10),
                temp=getattr(self, "gp_temp", 1.0),
                select_lambda=getattr(self, "gp_select_lambda", 0.5),
                use_knn=getattr(self, "gp_use_knn", True),
                use_select=getattr(self, "gp_use_select", True),
                use_cache=getattr(self, "gp_use_cache", False),
                cache_cap=getattr(self, "gp_cache_cap", 5),
                meta_layers=getattr(self, "gp_meta_layers", 1),
                meta_heads=getattr(self, "gp_meta_heads", 4),
                task_type=task_type_str,
            ).to(self.device)
        elif self.prompt_type == "MultiGprompt":
            nonlinearity = "prelu"
            self.Preprompt = NodePrePrompt(
                self.dataset_name,
                self.hid_dim,
                nonlinearity,
                0.9,
                0.9,
                0.1,
                0.001,
                1,
                0.3,
                self.device,
            ).to(self.device)
            self.Preprompt.load_state_dict(torch.load(self.pre_train_model_path))
            self.Preprompt.eval()
            self.feature_prompt = featureprompt(
                self.Preprompt.dgiprompt.prompt,
                self.Preprompt.graphcledgeprompt.prompt,
                self.Preprompt.lpprompt.prompt,
            ).to(self.device)
            dgiprompt = self.Preprompt.dgi.prompt
            graphcledgeprompt = self.Preprompt.graphcledge.prompt
            lpprompt = self.Preprompt.lp.prompt
            self.DownPrompt = downprompt(
                dgiprompt,
                graphcledgeprompt,
                lpprompt,
                0.001,
                self.hid_dim,
                self.output_dim,
                self.device,
            ).to(self.device)
        else:
            raise KeyError(" We don't support this kind of prompt.")

    def initialize_gnn(self):
        self.gnn = build_gnn(self.gnn_type, self.input_dim, self.hid_dim, self.num_layer)
        self.gnn.to(self.device)
        logger.info(self.gnn)

        if self.pre_train_model_path != "None" and self.prompt_type != "MultiGprompt":
            if self.gnn_type not in self.pre_train_model_path:
                raise ValueError(
                    f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model"
                )
            if self.dataset_name not in self.pre_train_model_path:
                raise ValueError(
                    f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset"
                )

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location="cpu"))
            self.gnn.to(self.device)
            logger.info("Successfully loaded pre-trained weights!")

    def return_pre_train_type(self, pre_train_model_path):
        valid_names = {
            "None",
            "DGI",
            "GraphMAE",
            "Edgepred_GPPT",
            "Edgepred_Gprompt",
            "GraphCL",
            "SimGRACE",
        }
        if pre_train_model_path == "None":
            return "None"
        # 约定文件名格式为 "<PreTrainType>.<gnn>.<hid_dim>hidden_dim.pth"
        head = os.path.basename(pre_train_model_path).split(".")[0]
        if head not in valid_names:
            raise ValueError(
                f"Cannot infer pre_train_type from path '{pre_train_model_path}': "
                f"the leading filename token '{head}' is not in {sorted(valid_names)}."
            )
        return head
