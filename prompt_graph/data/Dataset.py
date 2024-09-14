from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        初始化 GraphDataset
        :param graphs: 包含图对象的列表
        """
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        """
        返回数据集的大小
        :return: 数据集的大小
        """
        return len(self.graphs)

    def get(self, idx):
        """
        获取索引为 idx 的图
        :param idx: 索引
        :return: 图对象
        """
        graph = self.graphs[idx]
        # 可以在这里进行图数据的预处理或特征提取
        # 例如，如果每个图对象都有节点特征和边特征，可以返回它们
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph