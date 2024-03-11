
def GNNNodeEva(data, mask,  gnn, answering):
    gnn.eval()
    out = gnn(data.x, data.edge_index, batch=None)
    out = answering(out)
    pred = out.argmax(dim=1) 
    correct = pred[mask] == data.y[mask]  
    acc = int(correct.sum()) / int(mask.sum())  
    return acc


def GNNGraphEva(loader, gnn, answering, device):
    gnn.eval()
    if answering:
        answering.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        out = gnn(batch.x, batch.edge_index, batch.batch)
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  