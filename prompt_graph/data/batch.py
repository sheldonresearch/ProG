import torch
from torch_geometric.data import Data, Batch

class BatchFinetune(Data):
    r"""The Projected Randomized Block Coordinate Descent (PRBCD) adversarial
        attack from the `Robustness of Graph Neural Networks at Scale
        <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

        This attack uses an efficient gradient based approach that (during the
        attack) relaxes the discrete entries in the adjacency matrix
        :math:`\{0, 1\}` to :math:`[0, 1]` and solely perturbs the adjacency matrix
        (no feature perturbations). Thus, this attack supports all models that can
        handle weighted graphs that are differentiable w.r.t. these edge weights,
        *e.g.*, :class:`~torch_geometric.nn.conv.GCNConv` or
        :class:`~torch_geometric.nn.conv.GraphConv`. For non-differentiable models
        you might need modifications, e.g., see example for
        :class:`~torch_geometric.nn.conv.GATConv`.

        The memory overhead is driven by the additional edges (at most
        :attr:`block_size`). For scalability reasons, the block is drawn with
        replacement and then the index is made unique. Thus, the actual block size
        is typically slightly smaller than specified.

        This attack can be used for both global and local attacks as well as
        test-time attacks (evasion) and training-time attacks (poisoning). Please
        see the provided examples.

        This attack is designed with a focus on node- or graph-classification,
        however, to adapt to other tasks you most likely only need to provide an
        appropriate loss and model. However, we currently do not support batching
        out of the box (sampling needs to be adapted).

        .. note::
            For examples of using the PRBCD Attack, see
            `examples/contrib/rbcd_attack.py
            <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
            contrib/rbcd_attack.py>`_
            for a test time attack (evasion) or
            `examples/contrib/rbcd_attack_poisoning.py
            <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
            contrib/rbcd_attack_poisoning.py>`_
            for a training time (poisoning) attack.

        Args:
            model (torch.nn.Module): The GNN module to assess.
            block_size (int): Number of randomly selected elements in the
                adjacency matrix to consider.
            epochs (int, optional): Number of epochs (aborts early if
                :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
            epochs_resampling (int, optional): Number of epochs to resample the
                random block. (default: obj:`100`)
            loss (str or callable, optional): A loss to quantify the "strength" of
                an attack. Note that this function must match the output format of
                :attr:`model`. By default, it is assumed that the task is
                classification and that the model returns raw predictions (*i.e.*,
                no output activation) or uses :obj:`logsoftmax`. Moreover, and the
                number of predictions should match the number of labels passed to
                :attr:`attack`. Either pass a callable or one of: :obj:`'masked'`,
                :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
                (default: :obj:`'prob_margin'`)
            metric (callable, optional): Second (potentially
                non-differentiable) loss for monitoring or early stopping (if
                :obj:`mode='greedy'`). (default: same as :attr:`loss`)
            lr (float, optional): Learning rate for updating edge weights.
                Additionally, it is heuristically corrected for :attr:`block_size`,
                budget (see :attr:`attack`) and graph size. (default: :obj:`1_000`)
            is_undirected (bool, optional): If :obj:`True` the graph is
                assumed to be undirected. (default: :obj:`True`)
            log (bool, optional): If set to :obj:`False`, will not log any learning
                progress. (default: :obj:`True`)
        """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'center_node_idx']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchMasking(Data):
    r"""The Projected Randomized Block Coordinate Descent (PRBCD) adversarial
        attack from the `Robustness of Graph Neural Networks at Scale
        <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

        This attack uses an efficient gradient based approach that (during the
        attack) relaxes the discrete entries in the adjacency matrix
        :math:`\{0, 1\}` to :math:`[0, 1]` and solely perturbs the adjacency matrix
        (no feature perturbations). Thus, this attack supports all models that can
        handle weighted graphs that are differentiable w.r.t. these edge weights,
        *e.g.*, :class:`~torch_geometric.nn.conv.GCNConv` or
        :class:`~torch_geometric.nn.conv.GraphConv`. For non-differentiable models
        you might need modifications, e.g., see example for
        :class:`~torch_geometric.nn.conv.GATConv`.

        The memory overhead is driven by the additional edges (at most
        :attr:`block_size`). For scalability reasons, the block is drawn with
        replacement and then the index is made unique. Thus, the actual block size
        is typically slightly smaller than specified.

        This attack can be used for both global and local attacks as well as
        test-time attacks (evasion) and training-time attacks (poisoning). Please
        see the provided examples.

        This attack is designed with a focus on node- or graph-classification,
        however, to adapt to other tasks you most likely only need to provide an
        appropriate loss and model. However, we currently do not support batching
        out of the box (sampling needs to be adapted).

        .. note::
            For examples of using the PRBCD Attack, see
            `examples/contrib/rbcd_attack.py
            <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
            contrib/rbcd_attack.py>`_
            for a test time attack (evasion) or
            `examples/contrib/rbcd_attack_poisoning.py
            <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
            contrib/rbcd_attack_poisoning.py>`_
            for a training time (poisoning) attack.

        Args:
            model (torch.nn.Module): The GNN module to assess.
            block_size (int): Number of randomly selected elements in the
                adjacency matrix to consider.
            epochs (int, optional): Number of epochs (aborts early if
                :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
            epochs_resampling (int, optional): Number of epochs to resample the
                random block. (default: obj:`100`)
            loss (str or callable, optional): A loss to quantify the "strength" of
                an attack. Note that this function must match the output format of
                :attr:`model`. By default, it is assumed that the task is
                classification and that the model returns raw predictions (*i.e.*,
                no output activation) or uses :obj:`logsoftmax`. Moreover, and the
                number of predictions should match the number of labels passed to
                :attr:`attack`. Either pass a callable or one of: :obj:`'masked'`,
                :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
                (default: :obj:`'prob_margin'`)
            metric (callable, optional): Second (potentially
                non-differentiable) loss for monitoring or early stopping (if
                :obj:`mode='greedy'`). (default: same as :attr:`loss`)
            lr (float, optional): Learning rate for updating edge weights.
                Additionally, it is heuristically corrected for :attr:`block_size`,
                budget (see :attr:`attack`) and graph size. (default: :obj:`1_000`)
            is_undirected (bool, optional): If :obj:`True` the graph is
                assumed to be undirected. (default: :obj:`True`)
            log (bool, optional): If set to :obj:`False`, will not log any learning
                progress. (default: :obj:`True`)
        """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key  == 'masked_edge_idx':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0



class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)

                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)


                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
