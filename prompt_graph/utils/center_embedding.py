import torch



def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    c /= class_counts
    
    return c

def distance2center(input,center):
    n = input.size(0)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)

    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance