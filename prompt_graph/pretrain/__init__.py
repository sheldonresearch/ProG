from .DGI import DGI
from .Edgepred_GPPT import Edgepred_GPPT
from .Edgepred_Gprompt import Edgepred_Gprompt
from .GraphCL import GraphCL
from .GraphMAE import GraphMAE
from .MultiGPrompt import GraphPrePrompt, NodePrePrompt, prompt_pretrain_sample
from .SimGRACE import SimGRACE

__all__ = [
    "DGI",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
    "GraphCL",
    "GraphMAE",
    "GraphPrePrompt",
    "NodePrePrompt",
    "SimGRACE",
    "prompt_pretrain_sample",
]
