

def GPFEva(loader, gnn, prompt, answering, device):
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