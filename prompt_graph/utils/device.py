"""Canonical device-resolution helper.

This module centralizes device selection logic so that the rest of the codebase
does not repeat ``torch.cuda.is_available()`` triads.

Note: the previous ``PROG_USE_MPS=1`` environment-variable gate has been
removed in favor of an explicit ``--device mps`` CLI argument. To use Apple
Silicon GPUs, pass ``--device mps`` directly instead of toggling an env var.
"""

from __future__ import annotations

from typing import Union

import torch

DeviceSpec = Union[int, str, torch.device, None]


def _autodetect() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceSpec = None) -> torch.device:
    """Resolve a flexible device spec into a concrete ``torch.device``.

    Accepted forms:
      * ``torch.device`` -> returned as-is.
      * ``None`` or ``"auto"`` -> autodetect CUDA > MPS > CPU.
      * ``"cpu"`` -> CPU.
      * ``"mps"`` -> MPS if available, else CPU.
      * ``"cuda"`` / ``"cuda:N"`` -> the requested CUDA device.
      * ``int`` -> legacy behavior: ``cuda:N`` if CUDA available, else falls
        back via autodetect (MPS > CPU). Mirrors the long-standing inline
        ternary used across the codebase.

    Anything else raises ``ValueError``.
    """
    if isinstance(device, torch.device):
        return device

    if device is None:
        return _autodetect()

    # ``bool`` is an ``int`` subclass — reject early so ``True``/``False``
    # don't sneak through the int branch as ``cuda:1`` / ``cuda:0``.
    if isinstance(device, bool):
        raise ValueError(f"Unrecognized device spec: {device!r}")

    if isinstance(device, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return _autodetect()

    if isinstance(device, str):
        spec = device.strip().lower()
        if spec == "auto":
            return _autodetect()
        if spec == "cpu":
            return torch.device("cpu")
        if spec == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if spec == "cuda" or spec.startswith("cuda:"):
            return torch.device(spec)
        # argparse defaults --device to the string "0"; accept bare digits.
        if spec.lstrip("-").isdigit():
            return resolve_device(int(spec))

    raise ValueError(f"Unrecognized device spec: {device!r}")
