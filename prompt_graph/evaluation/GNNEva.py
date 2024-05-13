import torchmetrics
import torch

def GNNNodeEva(data, idx_test,  gnn, answering, num_class, device):
    gnn.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()

    out = gnn(data.x, data.edge_index, batch=None)
    out = answering(out)
    pred = out.argmax(dim=1) 

    acc = accuracy(pred[idx_test], data.y[idx_test])
    f1 = macro_f1(pred[idx_test], data.y[idx_test])
    roc = auroc(out[idx_test], data.y[idx_test]) 
    return acc.item(), f1.item(), roc.item()

def GNNGraphEva(loader, gnn, answering, num_class, device):
    gnn.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    if answering:
        answering.eval()
    with torch.no_grad(): 
        for index, batch in enumerate(loader): 
            batch = batch.to(device) 
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