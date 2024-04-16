def GPPTEva(data, idx_test, gnn, prompt):
    # gnn.eval()
    prompt.eval()
    node_embedding = gnn(data.x, data.edge_index)
    out = prompt(node_embedding, data.edge_index)
    pred = out.argmax(dim=1)  
    correct = pred[idx_test] == data.y[idx_test]  
    acc = int(correct.sum()) / len(idx_test)  
    return acc

def GPPTGraphEva(loader, gnn, prompt, device):
    # batch must be 1
    prompt.eval()
    correct = 0
    for batch in loader: 
        batch=batch.to(device)              
        node_embedding = gnn(batch.x,batch.edge_index)
        out = prompt(node_embedding, batch.edge_index)

        # 找到每个预测中概率最大的索引（类别）
        predicted_classes = out.argmax(dim=1)

        # 统计每个类别获得的票数
        votes = predicted_classes.bincount(minlength=out.shape[1])

        # 找出票数最多的类别
        final_class = votes.argmax().item()

        correct += int((final_class == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  