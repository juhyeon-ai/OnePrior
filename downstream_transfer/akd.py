import torch
import torch.nn.functional as F

from anchors import anchor_coordinates


def akd_weight(epoch, epochs, weight=0.3):
    return weight * max(0.0, 1.0 - (epoch - 1) / max(epochs // 2, 1))


def akd_loss(scores, prior_scores):
    target = torch.log_softmax(prior_scores.float(), dim=0)
    current = torch.log_softmax(scores.float(), dim=0)
    return (target.exp() * (target - current)).sum()


@torch.no_grad()
def prior_attention_scores(prior, features, anchor_features):
    return prior(anchor_coordinates(features, anchor_features))["scores"]


def train_epoch(model, prior, anchor_features, bags, labels, optimizer, epoch, epochs, weight=0.3, device="cpu"):
    model.train()
    prior.eval()
    beta = akd_weight(epoch, epochs, weight)
    total = 0.0
    for i in torch.randperm(len(bags)).tolist():
        features = bags[i].to(device)
        label = torch.tensor([labels[i]], device=device)
        output = model(features)
        loss = model.loss(output, label) if hasattr(model, "loss") else F.cross_entropy(output["logits"], label)
        if beta > 0:
            loss = loss + beta * akd_loss(output["scores"], prior_attention_scores(prior, features, anchor_features))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(bags)


def fit(model, prior, anchor_features, bags, labels, epochs=20, lr=1e-4, weight_decay=1e-5, min_lr=5e-6, weight=0.3, device="cpu"):
    model.to(device)
    prior.to(device).requires_grad_(False)
    anchor_features = anchor_features.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    history = []
    for epoch in range(1, epochs + 1):
        history.append(train_epoch(model, prior, anchor_features, bags, labels, optimizer, epoch, epochs, weight, device))
        scheduler.step()
    return history
