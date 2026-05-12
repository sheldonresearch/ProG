"""集中管理 ProG 仓库内所有路径常量与构造逻辑。

约定:
    REPO_ROOT 通过 ``Path(__file__).resolve().parents[2]`` 推导, 因此前提是
    本模块位于 ``<repo>/prompt_graph/utils/paths.py``, 即 "从仓库根目录运行"。
    如果包被安装到 site-packages, 该常量将指向 site-packages 中的某个目录,
    届时请使用下方的环境变量覆盖。

环境变量:
    ``PROG_DATA_ROOT``       覆盖 :data:`DATA_ROOT` (默认 ``<repo>/data``)。
    ``PROG_EXPERIMENT_ROOT`` 覆盖 :data:`EXPERIMENT_ROOT` (默认 ``<repo>/Experiment``)。
    ``PROG_OGB_ROOT``        覆盖 :func:`ogb_dataset_root` (默认 ``<DATA_ROOT>/OGB``)。
                             历史版本将 OGB 数据下载到 ``./dataset``, 升级后请
                             通过此变量指回旧目录, 否则会重新下载。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ProG/


def _env_or_default(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


DATA_ROOT = _env_or_default("PROG_DATA_ROOT", REPO_ROOT / "data")
EXPERIMENT_ROOT = _env_or_default("PROG_EXPERIMENT_ROOT", REPO_ROOT / "Experiment")


def induced_graph_dir(dataset_name: str) -> Path:
    return EXPERIMENT_ROOT / "induced_graph" / dataset_name


def sample_dir(task_type: str, shot_num: int, dataset_name: str) -> Path:
    """``Experiment/sample_data/<task_type>/<dataset_name>/<shot_num>_shot``.

    ``task_type`` 取值为 ``'Node'`` 或 ``'Graph'``。
    """
    return EXPERIMENT_ROOT / "sample_data" / task_type / dataset_name / f"{shot_num}_shot"


def excel_result_dir(task_type: str, shot_num: int, dataset_name: str) -> Path:
    return EXPERIMENT_ROOT / "ExcelResults" / task_type / f"{shot_num}shot" / dataset_name


def pretrained_model_dir(dataset_name: str) -> Path:
    return EXPERIMENT_ROOT / "pre_trained_model" / dataset_name


def tudataset_root() -> Path:
    return DATA_ROOT / "TUDataset"


def ogb_dataset_root() -> Path:
    """OGB 数据集的根目录, 默认为 ``<DATA_ROOT>/OGB``。

    旧版本默认 ``./dataset``; 如需沿用, 请设置 ``PROG_OGB_ROOT=./dataset``。
    """
    override = os.environ.get("PROG_OGB_ROOT")
    if override is not None:
        return Path(override)
    return DATA_ROOT / "OGB"


def planetoid_root() -> Path:
    return DATA_ROOT / "Planetoid"
