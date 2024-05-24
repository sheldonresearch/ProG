import torch
import torchmetrics


def MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, Preprompt, DownPrompt, sp_adj, num_class, device):
    Preprompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    with torch.no_grad(): 
        embeds1, _ = Preprompt.embed(prompt_feature, sp_adj, True, None, False)
        test_embs1 = embeds1[0, idx_test]
        print('idx_test', idx_test)
        logits = DownPrompt(test_embs, test_embs1, test_lbls)


        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, test_lbls)
        f1 = macro_f1(preds, test_lbls)
        roc = auroc(logits, test_lbls) 
        prc = auprc(logits, test_lbls) 
    return acc.item(), f1.item(), roc.item(), prc.item()

def MultiGpromptGraphEva(test_embs, test_lbls, centroids_embeds, batch, DownPrompt, num_class, device):
    
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    with torch.no_grad():
        graph_embeds = DownPrompt.graph_forward(test_embs, batch)
        ret = torch.nn.functional.cosine_similarity(graph_embeds.unsqueeze(1), centroids_embeds.unsqueeze(0), dim=-1)
        logits = torch.nn.functional.log_softmax(ret, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        acc = accuracy(preds, test_lbls)
        f1 = macro_f1(preds, test_lbls)
        roc = auroc(logits, test_lbls) 
        prc = auprc(logits, test_lbls) 

    return acc.item(), f1.item(), roc.item(), prc.item()