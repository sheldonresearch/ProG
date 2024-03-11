def GPFNodeEva(data, mask,  gnn, prompt, answering, device):
    gnn.eval()
    data.x = prompt.add(data.x)
    out = gnn(data.x, data.edge_index, batch=None)
    out = answering(out)
    pred = out.argmax(dim=1) 
    correct = pred[mask] == data.y[mask]  
    acc = int(correct.sum()) / int(mask.sum())  
    return acc

            

def GPFGraphEva( loader, gnn, prompt, answering, device):
    prompt.eval()
    if answering:
        answering.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        batch.x = prompt.add(batch.x)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  