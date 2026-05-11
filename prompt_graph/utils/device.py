"""Centralized device resolution.

调用方既可以传 int（用于 CUDA 选卡），也可以传 torch.device。
兜底顺序：CUDA → MPS (需 PROG_USE_MPS=1) → CPU。

把这套逻辑集中在一处，避免 bench.py / downstream_task.py / BaseTask 各算各的，
然后在不同分支之间漂移。
"""
import os

import torch


def resolve_device(device):
    """Return a torch.device based on input device (int or torch.device).

    - 如果传入已经是 torch.device，直接返回。
    - 否则把它当 int，按顺序探测 CUDA/MPS/CPU。
    """
    if isinstance(device, torch.device):
        return device
    if torch.cuda.is_available():
        return torch.device(f'cuda:{int(device)}')
    if os.environ.get('PROG_USE_MPS') == '1' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
