import argparse

from prompt_graph.data import load4graph, load4node, load_induced_graphs
from prompt_graph.tasker import GraphTask, NodeTask
from prompt_graph.utils import (
    apply_log_level,
    get_args,
    get_logger,
    resolve_device,
    seed_everything,
)

logger = get_logger(__name__)


def get_downstream_task_delegate(args: argparse.Namespace):

    seed_everything(args.seed)
    runtime_device = resolve_device(args.device)

    if args.downstream_task == "NodeTask":
        data, input_dim, output_dim = load4node(args.dataset_name)
        data = data.to(runtime_device)
        if args.prompt_type in ["Gprompt", "All-in-one", "GPF", "GPF-plus"]:
            graphs_list = load_induced_graphs(args.dataset_name, data, runtime_device)
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
