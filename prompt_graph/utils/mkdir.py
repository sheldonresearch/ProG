import os

from .logging import get_logger

logger = get_logger(__name__)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        logger.info(f"create folder {path}")
    else:
        logger.info(f"folder exists! {path}")
