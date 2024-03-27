data
=======================

.. contents:: Contents
    :local:


batch
-------------------

.. currentmodule:: prompt_graph.data.batch


.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/inherited_class.rst

    BatchFinetune
    BatchMasking
    BatchAE
    BatchSubstructContext



dataloader
-------------------

.. currentmodule::  prompt_graph.data.dataloader


.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: autosummary/inherited_class.rst

    DataLoaderFinetune
    DataLoaderMasking
    DataLoaderAE
    DataLoaderSubstructContext

graph_split
-------------------
.. currentmodule:: prompt_graph.data.graph_split


.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/inherited_class.rst

    graph_split

induced_graphs
-------------------

.. currentmodule:: prompt_graph.data.induced_graph


.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: autosummary/inherited_class.rst

    induced_graphs
    split_induced_graphs
    multi_class_NIG
    nodes_split
    edge_split
    induced_graphs_nodes
    induced_graphs_edges
    induced_graphs_graphs
    induced_graph_2_K_shot
    load_tasks



load4data
-------------------

.. currentmodule:: prompt_graph.data.load4data


.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/inherited_class.rst

    load4graph
    load4node
    load4link_prediction_single_graph
    load4link_prediction_multi_graph
    NodePretrain

loader
-------------------

.. currentmodule:: prompt_graph.data.loader


.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/inherited_class.rst

    nx_to_graph_data_obj
    graph_data_obj_to_nx
    BioDataset


pooling
-------------------

.. currentmodule:: prompt_graph.data.pooling


.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/inherited_class.rst

    topk
    filter_adj
    TopKPoolings
    SAGPooling
