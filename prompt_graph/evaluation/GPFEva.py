import torchmetrics

def GPFEva(loader, gnn, prompt, answering, num_class, device):
    prompt.eval()
    if answering:
        answering.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()

    for batch in loader: 
        batch = batch.to(device) 
        batch.x = prompt.add(batch.x)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        if answering:
            out = answering(out)  
        pred = out.argmax(dim=1)  

        acc = accuracy(pred, batch.y)
        ma_f1 = macro_f1(pred, batch.y)
        roc = auroc(out, batch.y) 
        # print(acc)
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
       
    return acc.item(), ma_f1.item(), roc.item()