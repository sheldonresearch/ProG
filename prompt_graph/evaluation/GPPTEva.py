def GPPTEva(data, idx_test, gnn, prompt):
    # gnn.eval()
    prompt.eval()
    node_embedding = gnn(data.x, data.edge_index)
    out = prompt(node_embedding, data.edge_index)
    pred = out.argmax(dim=1)  
    correct = pred[idx_test] == data.y[idx_test]  
    acc = int(correct.sum()) / len(idx_test)  
    return acc