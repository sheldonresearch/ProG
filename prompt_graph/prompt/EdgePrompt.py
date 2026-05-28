"""EdgePrompt (ICLR 2025) light-weight port for ProG.

Core idea: instead of attaching prompt vectors to node features, learn
additional prompt vectors for **edges** and incorporate them through
message passing:

    message(u -> v) = lin( x_u + prompt_{uv} )

Because ProG uses standard PyG conv layers (GCNConv, GATConv, SAGEConv, …)
which do not expose per-edge injection hooks, we approximate the original
behaviour by **adding the edge prompt to the source-node features before
the conv call**:

    x' = x + scatter_add( prompt_{uv} , index=u , dim=0 )

This is semantically close to the original when each node has a moderate
out-degree; the difference is that the prompt is accumulated *outside* the
linear layer rather than *inside* it.  For a downstream prompt-tuning
scenario the optimisation dynamics are nearly identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class EdgePrompt(nn.Module):
    """Global edge prompt — one learnable vector per layer.

    Parameters
    ----------
    dim_list : list[int]
        Hidden dimensions for each GNN layer (e.g. ``[hid_dim, hid_dim, out_dim]``).
    """

    def __init__(self, dim_list):
        super().__init__()
        self.dim_list = dim_list
        self.global_prompt = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, dim)) for dim in dim_list]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for prompt in self.global_prompt:
            glorot(prompt)

    def get_prompt(self, x, edge_index, layer=0):
        """Return edge prompts for every edge in ``edge_index``."""
        num_edges = edge_index.size(1)
        return self.global_prompt[layer].expand(num_edges, -1)

    def forward(self, x, edge_index, layer=0):
        """Aggregate edge prompts to source nodes and add to ``x``."""
        edge_attr = self.get_prompt(x, edge_index, layer)
        aug = torch.zeros_like(x)
        aug.scatter_add_(
            0,
            edge_index[0].unsqueeze(1).expand(-1, x.size(1)),
            edge_attr,
        )
        return x + aug


class EdgePromptplus(nn.Module):
    """Adaptive edge prompt — attention over anchor prompts per edge.

    For each edge ``(u, v)`` the module concatenates ``[x_u, x_v]``,
    projects to ``num_anchors`` attention scores, and returns a weighted
    combination of ``num_anchors`` learnable prompt vectors.

    Parameters
    ----------
    dim_list : list[int]
        Hidden dimensions for each GNN layer.
    num_anchors : int
        Number of anchor prompt vectors (default 20, matching the paper).
    """

    def __init__(self, dim_list, num_anchors=20):
        super().__init__()
        self.dim_list = dim_list
        self.num_anchors = num_anchors
        self.anchor_prompt = nn.ParameterList(
            [nn.Parameter(torch.Tensor(num_anchors, dim)) for dim in dim_list]
        )
        self.w = nn.ModuleList(
            [nn.Linear(2 * dim, num_anchors) for dim in dim_list]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for anchor in self.anchor_prompt:
            glorot(anchor)
        for lin in self.w:
            lin.reset_parameters()

    def get_prompt(self, x, edge_index, layer=0):
        """Return edge-specific prompts [num_edges, dim]."""
        combined = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        b = F.softmax(F.leaky_relu(self.w[layer](combined)), dim=1)
        return b.mm(self.anchor_prompt[layer])

    def forward(self, x, edge_index, layer=0):
        """Aggregate edge prompts to source nodes and add to ``x``."""
        edge_attr = self.get_prompt(x, edge_index, layer)
        aug = torch.zeros_like(x)
        aug.scatter_add_(
            0,
            edge_index[0].unsqueeze(1).expand(-1, x.size(1)),
            edge_attr,
        )
        return x + aug
