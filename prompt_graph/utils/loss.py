import torch
import torch.nn.functional as F
import torch.nn as nn

class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Gprompt_tuning_loss, self).__init__()
        self.tau = tau
    
    def forward(self, embedding, center_embedding, labels):
        # 计算所有样本与两个类原型的相似度
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1) / self.tau
        # embedding.unsqueeze(1) 的大小变为 [batch_size, 1, 128]
        # center_embedding.unsqueeze(0) 的大小变为 [1, label_num, 128]
        # 为每个样本选择其真实类别的相似度
        true_class_sim = similarity_matrix[torch.arange(embedding.size(0)), labels]

        # 计算分母（对每个样本，包括所有类的相似度）
        all_classes_sim = similarity_matrix.logsumexp(dim=1)

        loss = - (true_class_sim - all_classes_sim).mean()

        return loss

def Gprompt_link_loss(node_emb, pos_emb, neg_emb, temperature=0.2):
    r"""Refer to GraphPrompt original codes"""
    x = torch.exp(F.cosine_similarity(node_emb, pos_emb, dim=-1) / temperature)
    y = torch.exp(F.cosine_similarity(node_emb, neg_emb, dim=-1) / temperature)

    loss = -1 * torch.log(x / (x + y) )
    return loss.mean()