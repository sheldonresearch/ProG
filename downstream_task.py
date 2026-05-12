import argparse
import os
import pickle

from prompt_graph.data import induced_graph_cache_path, load4graph, load4node, split_induced_graphs
from prompt_graph.tasker import GraphTask, NodeTask
from prompt_graph.utils import (
    apply_log_level,
    get_args,
    get_logger,
    induced_graph_dir,
    resolve_device,
    seed_everything,
)

logger = get_logger(__name__)


def load_induced_graph(dataset_name, data, device):

    folder_path = str(induced_graph_dir(dataset_name))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_mask = getattr(data, "train_mask", None)
    file_path = induced_graph_cache_path(
        folder_path,
        smallest_size=100,
        largest_size=300,
        leak_safe=train_mask is not None,
    )
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            logger.info("loading induced graph...")
            graphs_list = pickle.load(f)
            logger.info("Done!!!")
    else:
        logger.info("Begin split_induced_graphs.")
        split_induced_graphs(
            data,
            folder_path,
            device,
            smallest_size=100,
            largest_size=300,
            train_mask=train_mask,
        )
        with open(file_path, "rb") as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


def get_downstream_task_delegate(args: argparse.Namespace):

    seed_everything(args.seed)
    runtime_device = resolve_device(args.device)

    if args.downstream_task == "NodeTask":
        data, input_dim, output_dim = load4node(args.dataset_name)
        data = data.to(runtime_device)
        if args.prompt_type in ["Gprompt", "All-in-one", "GPF", "GPF-plus"]:
            graphs_list = load_induced_graph(args.dataset_name, data, runtime_device)
        else:
            graphs_list = None
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
            lr=args.lr,
            wd=args.decay,
            batch_size=args.batch_size,
            data=data,
            input_dim=input_dim,
            output_dim=output_dim,
            graphs_list=graphs_list,
        )

    elif args.downstream_task == "GraphTask":
        input_dim, output_dim, dataset = load4graph(args.dataset_name)

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
            lr=args.lr,
            wd=args.decay,
            batch_size=args.batch_size,
            dataset=dataset,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unexpected args.downstream_task type {args.downstream_task}.")

    return tasker


if __name__ == "__main__":
    args = get_args()
    apply_log_level(args.log_level, args.quiet)
    logger.info("dataset_name %s", args.dataset_name)

    tasker = get_downstream_task_delegate(args=args)

    _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _ = tasker.run()

    print(f"Final Accuracy {test_acc:.4f}±{std_test_acc:.4f}(std)")
    print(f"Final F1 {f1:.4f}±{std_f1:.4f}(std)")
    print(f"Final AUROC {roc:.4f}±{std_roc:.4f}(std)")

    pre_train_type = tasker.pre_train_type
