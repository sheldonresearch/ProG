"""Centralized logger factory.

Future work: migrate 160+ scattered print() calls in tasker/, pretrain/ to
`logger = get_logger(__name__)` + `logger.info(...)`. Out of scope for this
unit (which only introduces the utility).

Log level can be controlled via the `PROG_LOG_LEVEL` environment variable
(default: INFO). Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.

Usage:
    from prompt_graph.utils import get_logger
    logger = get_logger(__name__)
    logger.info('training started')
    logger.warning('using mock dataset')
"""

import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_once() -> None:
    """Configure the root prompt_graph logger exactly once.

    Idempotent: re-importing or calling get_logger() multiple times does not
    add duplicate handlers.
    """
    global _configured
    if _configured:
        return
    root = logging.getLogger("prompt_graph")
    level_name = os.environ.get("PROG_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.propagate = False  # avoid double-logging if user has root logger configured
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the prompt_graph namespace.

    If `name` doesn't already start with 'prompt_graph', it is prefixed so all
    project logs share a parent and inherit configuration.
    """
    _configure_root_once()
    if not name.startswith("prompt_graph"):
        name = f"prompt_graph.{name}"
    return logging.getLogger(name)


def apply_log_level(log_level: Optional[str], quiet: bool = False) -> None:
    """Apply a runtime log level to the prompt_graph logger.

    Intended for use by CLI entrypoints: parse --log-level / --quiet and call
    this once after argparse. `quiet` overrides `log_level` and forces WARNING.
    """
    _configure_root_once()
    root = logging.getLogger("prompt_graph")
    if quiet:
        root.setLevel(logging.WARNING)
        return
    if log_level:
        root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
