

def GPPTEva(data, mask, gnn, prompt):
    # gnn.eval()
    prompt.eval()
    node_embedding = gnn(data.x, data.edge_index)
    out = prompt(node_embedding, data.edge_index)
    pred = out.argmax(dim=1)  
    correct = pred[mask] == data.y[mask]  
    acc = int(correct.sum()) / int(mask.sum())  
    return acc