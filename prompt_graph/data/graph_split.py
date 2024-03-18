import random


def graph_split(graph_list, shot_num):
# 分类并选择每个类别的图
    class_datasets = {}
    for data in graph_list:
        label = data.y
        if label not in class_datasets:
            class_datasets[label] = []
        class_datasets[label].append(data)

    train_data = []
    remaining_data = []
    for label, data_list in class_datasets.items():
        train_data.extend(data_list[:shot_num])
        random.shuffle(train_data)
        remaining_data.extend(data_list[shot_num:])

    # 将剩余的数据 1：9 划分为测试集和验证集
    random.shuffle(remaining_data)
    val_dataset_size = len(remaining_data) // 9
    val_dataset = remaining_data[:val_dataset_size]
    test_dataset = remaining_data[val_dataset_size:]
    return train_data, test_dataset, val_dataset
