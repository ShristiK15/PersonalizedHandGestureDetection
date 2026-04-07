import torch
import torch.nn.functional as F

def compute_prototypes(embeddings, labels, n_way):
    prototypes = []
    for i in range(n_way):
        proto = embeddings[labels == i].mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)

def proto_loss(model, support_x, support_y, query_x, query_y, n_way, temperature=10.0):
    # embeddings
    support_emb = model(support_x)
    query_emb   = model(query_x)

    # prototypes
    prototypes = compute_prototypes(support_emb, support_y, n_way)

    # cosine similarity
    query_emb = F.normalize(query_emb, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    logits = torch.matmul(query_emb, prototypes.T)

    # ==========================================
    # THE FIX: Scale the logits to prevent catastrophic forgetting
    # ==========================================
    logits = logits * temperature

    loss = F.cross_entropy(logits, query_y)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_y).float().mean()

    return loss, acc