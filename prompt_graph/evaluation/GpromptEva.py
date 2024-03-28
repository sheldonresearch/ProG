import torch.nn.functional as F

def GpromptEva(loader, gnn, prompt, center_embedding, device):
    prompt.eval()
    correct = 0
    for batch in loader: 
        batch = batch.to(device) 
        out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'Gprompt')
        similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
        pred = similarity_matrix.argmax(dim=1)
        correct += int((pred == batch.y).sum())  
    acc = correct / len(loader.dataset)
    return acc  