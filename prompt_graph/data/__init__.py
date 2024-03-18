from .load4data import load4graph, load4link_prediction_single_graph, load4node, load4link_prediction_multi_graph, NodePretrain
from .induced_graph import induced_graphs, split_induced_graphs
from . graph_split import graph_split
data_classes = [
    'DataLoaderFinetune',
    'DataLoaderMasking',
    'DataLoaderAE',
    'DataLoaderSubstructContext',
    'BatchFinetune',
    'BatchMasking',
    'BatchAE',
    'BioDataset',
]
