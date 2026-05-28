import argparse
import os
import random

import numpy as np
import pandas as pd

from prompt_graph.data import load4graph, load4node, load_induced_graphs
from prompt_graph.tasker import GraphTask, NodeTask
from prompt_graph.utils import (
    apply_log_level,
    excel_result_dir,
    get_args,
    get_logger,
    resolve_device,
    seed_everything,
)
from prompt_graph.utils.report_data import ConfigBenchResult

logger = get_logger(__name__)


def get_runtime_device(device_id):
    return resolve_device(device_id)


"""
Auto bench function. Using predefined param grid to search best results
for 1 pretrained model.
You need to provide at least 3 arguments
pretrain_task,
dataset_name,
prompt_type

"""


def do_config_bench(args: argparse.Namespace):
    seed_everything(args.seed)
    runtime_device = get_runtime_device(args.device)

    # YAML/CLI override wins; otherwise fall back to dataset-specific defaults.
    param_grid = getattr(args, "param_grid", None)
    if param_grid is None:
        param_grid = {
            "learning_rate": 10 ** np.linspace(-3, -1, 1000),
            "weight_decay": 10 ** np.linspace(-5, -6, 1000),
            "batch_size": [32, 64, 128],
        }
        if args.dataset_name in ["ogbn-arxiv", "Flickr"]:
            # 大图数据集单 run 即可：random search 也只跑 1 次 (num_iter=1)，
            # 用显式列表表达"这里没有 grid"，避免 np.linspace 退化成重复值。
            param_grid = {
                "learning_rate": [1e-2],
                "weight_decay": [1e-5],
                "batch_size": [512],
            }

    logger.info("args.dataset_name %s", args.dataset_name)

    num_iter = getattr(args, "num_iter", None)
    if num_iter is None:
        num_iter = 10
        # Define special num_iter cases
        if args.prompt_type in ["MultiGprompt", "GPPT"]:
            logger.info("num_iter = 1")
            num_iter = 1
        if args.dataset_name in ["ogbn-arxiv", "Flickr"]:
            logger.info("num_iter = 1")
            num_iter = 1
    best_params = {}
    best_loss = float("inf")
    final_acc_mean = 0
    final_acc_std = 0
    final_f1_mean = 0
    final_f1_std = 0
    final_roc_mean = 0
    final_roc_std = 0

    # args.pretrain_task = 'GraphTask'
    # # # # # args.prompt_type = 'MultiGprompt'
    # args.dataset_name = 'COLLAB'
    # # args.dataset_name = 'Cora'
    # # num_iter = 1
    # args.shot_num = 1
    # args.pre_train_model_path='./Experiment/pre_trained_model/DD/DGI.GCN.128hidden_dim.pth'

    if args.pretrain_task == "NodeTask":
        data, input_dim, output_dim = load4node(args.dataset_name)
        data = data.to(runtime_device)
        if args.prompt_type in ["Gprompt", "All-in-one", "GPF", "GPF-plus"]:
            graphs_list = load_induced_graphs(args.dataset_name, data, runtime_device)
        else:
            graphs_list = None

    if args.pretrain_task == "GraphTask":
        input_dim, output_dim, dataset = load4graph(args.dataset_name)

    logger.info("num_iter %s", num_iter)
    for a in range(num_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        logger.info("params: %s", params)

        if args.pretrain_task == "NodeTask":
            tasker = NodeTask(
                pre_train_model_path=args.pre_train_model_path,
                dataset_name=args.dataset_name,
                num_layer=args.num_layer,
                gnn_type=args.gnn_type,
                hid_dim=args.hid_dim,
                prompt_type=args.prompt_type,
                epochs=args.epochs,
                shot_num=args.shot_num,
                device=runtime_device,
                lr=params["learning_rate"],
                wd=params["weight_decay"],
                batch_size=int(params["batch_size"]),
                data=data,
                input_dim=input_dim,
                output_dim=output_dim,
                graphs_list=graphs_list,
            )

        elif args.pretrain_task == "GraphTask":
            tasker = GraphTask(
                pre_train_model_path=args.pre_train_model_path,
                dataset_name=args.dataset_name,
                num_layer=args.num_layer,
                gnn_type=args.gnn_type,
                hid_dim=args.hid_dim,
                prompt_type=args.prompt_type,
                epochs=args.epochs,
                shot_num=args.shot_num,
                device=runtime_device,
                lr=params["learning_rate"],
                wd=params["weight_decay"],
                batch_size=int(params["batch_size"]),
                dataset=dataset,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        else:
            raise ValueError(f"Unexpected pretrain_task: {args.pretrain_task}.")
        pre_train_type = tasker.pre_train_type

        # 返回平均损失
        (
            avg_best_loss,
            mean_test_acc,
            std_test_acc,
            mean_f1,
            std_f1,
            mean_roc,
            std_roc,
            mean_prc,
            std_prc,
        ) = tasker.run()

        # Convert each metric to Python float
        avg_best_loss = float(avg_best_loss)
        mean_test_acc = float(mean_test_acc)
        std_test_acc = float(std_test_acc)
        mean_f1 = float(mean_f1)
        std_f1 = float(std_f1)
        mean_roc = float(mean_roc)
        std_roc = float(std_roc)
        mean_prc = float(mean_prc)
        std_prc = float(std_prc)
        logger.info(
            "For %sth searching, Tested Params: %s, Avg Best Loss: %s", a, params, avg_best_loss
        )

        if avg_best_loss < best_loss:
            best_loss = avg_best_loss
            best_params = params
            final_acc_mean = mean_test_acc
            final_acc_std = std_test_acc
            final_f1_mean = mean_f1
            final_f1_std = std_f1
            final_roc_mean = mean_roc
            final_roc_std = std_roc

    if isinstance(best_params, dict):
        best_params = {k: float(v) for k, v in best_params.items()}
    return ConfigBenchResult(
        pretrain_task_type=args.pretrain_task,
        pre_train_type=pre_train_type,
        dataset_name=args.dataset_name,
        prompt_type=args.prompt_type,
        best_params=best_params,
        best_loss=best_loss,
        final_acc_mean=final_acc_mean,
        final_acc_std=final_acc_std,
        final_f1_mean=final_f1_mean,
        final_f1_std=final_f1_std,
        final_roc_mean=final_roc_mean,
        final_roc_std=final_roc_std,
    )


# pre_train_types = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
# prompt_types = ['None', 'GPPT', 'All-in-one', 'Gprompt', 'GPF', 'GPF-plus']
if __name__ == "__main__":
    args = get_args()
    apply_log_level(args.log_level, args.quiet)

    cbr_result = do_config_bench(args=args)

    file_name = args.gnn_type + "_total_results.xlsx"
    if args.pretrain_task == "NodeTask":
        file_path = os.path.join(
            str(excel_result_dir("Node", args.shot_num, args.dataset_name)), file_name
        )
    if args.pretrain_task == "GraphTask":
        file_path = os.path.join(
            str(excel_result_dir("Graph", args.shot_num, args.dataset_name)), file_name
        )
    data = pd.read_excel(file_path, index_col=0)

    col_name = f"{cbr_result.pre_train_type}+{args.prompt_type}"
    logger.info("col_name %s", col_name)
    data.at["Final Accuracy", col_name] = (
        f"{cbr_result.final_acc_mean:.4f}±{cbr_result.final_acc_std:.4f}"
    )
    data.at["Final F1", col_name] = f"{cbr_result.final_f1_mean:.4f}±{cbr_result.final_f1_std:.4f}"
    data.at["Final AUROC", col_name] = (
        f"{cbr_result.final_roc_mean:.4f}±{cbr_result.final_roc_std:.4f}"
    )
    data.to_excel(file_path)

    print("Data saved to " + file_path + " successfully.")

    print(
        f"After searching, Final Accuracy {cbr_result.final_acc_mean:.4f}±{cbr_result.final_acc_std:.4f}(std)"
    )
    print(
        f"After searching, Final F1 {cbr_result.final_f1_mean:.4f}±{cbr_result.final_f1_std:.4f}(std)"
    )
    print(
        f"After searching, Final AUROC {cbr_result.final_roc_mean:.4f}±{cbr_result.final_roc_std:.4f}(std)"
    )
    print("best_params ", cbr_result.best_params)
    print("best_loss ", cbr_result.best_loss)
