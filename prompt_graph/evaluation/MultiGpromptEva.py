import torch



def MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, Preprompt, DownPrompt, sp_adj):
    embeds1, _ = Preprompt.embed(prompt_feature, sp_adj, True, None, False)
    test_embs1 = embeds1[0, idx_test]
    print('idx_test', idx_test)
    logits = DownPrompt(test_embs, test_embs1, test_lbls)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    return acc.cpu().numpy()