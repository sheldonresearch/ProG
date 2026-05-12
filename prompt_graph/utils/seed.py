import os
import numpy as np
import random
import torch
from random import shuffle

from .logging import get_logger

logger = get_logger(__name__)



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def __seeds_list__(nodes):
    split_size = max(5, int(nodes.shape[0] / 400))
    seeds_list = list(torch.split(nodes, split_size))
    if len(seeds_list) < 400:
        logger.warning('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
        seeds_list = []
        while len(seeds_list) < 400:
            split_size = random.randint(3, 5)
            seeds_list_1 = torch.split(nodes, split_size)
            seeds_list = seeds_list + list(seeds_list_1)
            nodes = nodes[torch.randperm(nodes.shape[0])]
    shuffle(seeds_list)
    seeds_list = seeds_list[0:400]

    return seeds_list
