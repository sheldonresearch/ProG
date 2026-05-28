import argparse
import logging

logger = logging.getLogger(__name__)


def _build_parser():
    """Build the argparse parser.

    All arguments use ``argparse.SUPPRESS`` so that only flags explicitly
    passed on the command line appear in the resulting namespace. This lets
    us merge CLI > YAML > defaults cleanly in ``get_args``.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args still override YAML values)",
    )
    parser.add_argument("--pretrain_task", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--downstream_task", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=argparse.SUPPRESS,
        help="Choose the dataset of pretrainor downstream task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Device: int (legacy CUDA index), or auto/cuda:N/mps/cpu",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default=argparse.SUPPRESS,
        help="We support gnn like GCN GAT GT GCov GIN GraphSAGE, please read ProG.model module",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default=argparse.SUPPRESS,
        help="Choose the prompt type for node or graph task, for node task,we support GPPT, All-in-one, Gprompt for graph task , All-in-one, Gprompt, GPF, GPF-plus ",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=argparse.SUPPRESS,
        help="hideen layer of GNN dimensions (default: 128)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument("--shot_num", type=int, default=argparse.SUPPRESS, help="Number of shots")
    parser.add_argument(
        "--pre_train_model_path",
        type=str,
        default=argparse.SUPPRESS,
        help="add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.",
    )
    parser.add_argument(
        "--lr", type=float, default=argparse.SUPPRESS, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--decay", type=float, default=argparse.SUPPRESS, help="Weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of GNN message passing layers (default: 2).",
    )
    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Dropout ratio (default: 0.5)",
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default=argparse.SUPPRESS,
        help="Graph level pooling (sum, mean, max, set2set, attention)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default=argparse.SUPPRESS,
        help="How the node features across layers are combined. last, sum, max or concat",
    )
    parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS, help="Seed for splitting dataset."
    )
    parser.add_argument(
        "--runseed", type=int, default=argparse.SUPPRESS, help="Seed for running experiments."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of workers for dataset loading",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=argparse.SUPPRESS,
        help="A range of [1,2,3]-layer MLPs with equal width",
    )
    parser.add_argument(
        "--pnum",
        type=int,
        default=argparse.SUPPRESS,
        help="The number of independent basis for GPF-plus",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=argparse.SUPPRESS,
        help="The number of tasks for computing the mean metrices",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set root logger level (default: INFO)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence all INFO logs (equivalent to --log-level WARNING)",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of random-search trials in bench.py (default: 10, or 1 for large datasets)",
    )
    return parser


def _load_yaml_config(path):
    """Load YAML config from ``path``. Returns an empty dict if path is falsy."""
    if not path:
        return {}
    import yaml  # imported lazily so import-time has no hard dep when unused

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"YAML config at {path} must define a mapping at the top level, got {type(cfg).__name__}"
        )
    return cfg


def get_args():
    parser = _build_parser()
    args_ns = parser.parse_args()
    cli_overrides = vars(args_ns)
    yaml_cfg = _load_yaml_config(cli_overrides.pop("config", None))
    merged = {**DEFAULT_ARG_DICT, **yaml_cfg, **cli_overrides}
    return argparse.Namespace(**merged)


def get_args_by_call(
    pretrain_task: str = None,
    downstream_task: str = None,
    dataset_name: str = "Cora",
    device: int = 0,
    gnn_type: str = "GCN",
    prompt_type: str = "None",
    hid_dim: int = 128,
    batch_size: int = 128,
    epochs: int = 1000,
    shot_num: int = 1,
    pre_train_model_path: str = "None",
    lr: float = 0.001,
    decay: float = 0,
    num_layer: int = 2,
    dropout_ratio: float = 0.5,
    graph_pooling: str = "mean",
    JK: str = "last",
    seed: int = 42,
    runseed: int = 0,
    num_workers: int = 0,
    num_layers: int = 1,
    pnum: int = 5,
    task_num: int = 5,
    num_iter: int = None,
    log_level: str = "INFO",
    quiet: bool = False,
    **kwargs,
) -> argparse.Namespace:
    if len(kwargs) > 0:
        logger.warning(f"Warning! Unexpected argument input: {list(kwargs.keys())}")
    return argparse.Namespace(
        pretrain_task=pretrain_task,
        downstream_task=downstream_task,
        dataset_name=dataset_name,
        device=device,
        gnn_type=gnn_type,
        prompt_type=prompt_type,
        hid_dim=hid_dim,
        batch_size=batch_size,
        epochs=epochs,
        shot_num=shot_num,
        pre_train_model_path=pre_train_model_path,
        lr=lr,
        decay=decay,
        num_layer=num_layer,
        dropout_ratio=dropout_ratio,
        graph_pooling=graph_pooling,
        JK=JK,
        seed=seed,
        runseed=runseed,
        num_workers=num_workers,
        num_layers=num_layers,
        pnum=pnum,
        task_num=task_num,
        num_iter=num_iter,
        log_level=log_level,
        quiet=quiet,
        **kwargs,
    )


DEFAULT_ARG_DICT = dict(vars(get_args_by_call()))
DEFAULT_ARG_KEYS = list(DEFAULT_ARG_DICT.keys())
