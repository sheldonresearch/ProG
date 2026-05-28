import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb


class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super().__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
