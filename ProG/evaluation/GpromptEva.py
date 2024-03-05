
def GpromptEva(self, loader, gnn, prompt, answering):
    prompt.eval()
    if answering:
        answering.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(self.device) 
        out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'Gprompt')
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  