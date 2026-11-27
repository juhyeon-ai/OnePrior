import random

import numpy as np
import torch
import torch.nn as nn

from .common import GatedAttention, attention_pool, init_weights


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.ReLU(), nn.Linear(dim, dim, bias=False), nn.ReLU())

    def forward(self, h):
        return h + self.body(h)


class DTFDMIL(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=512, attn_dim=128, num_residuals=4, dropout=0.25,
                 num_groups=8, instances_per_group=1, selection="MaxMinS", bag_weight=0.7):
        super().__init__()
        assert selection in ("MaxMinS", "MaxS", "AFS") and instances_per_group >= 1
        self.stem = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.residual = nn.Sequential(*[ResidualBlock(embed_dim) for _ in range(num_residuals)])
        self.tier1_attention = GatedAttention(embed_dim, attn_dim, dropout)
        self.tier1_dropout = nn.Dropout(dropout)
        self.tier1_classifier = nn.Linear(embed_dim, num_classes)
        self.tier2_attention = GatedAttention(embed_dim, attn_dim)
        self.tier2_classifier = nn.Linear(embed_dim, num_classes)
        self.num_groups, self.instances_per_group, self.selection, self.bag_weight = num_groups, instances_per_group, selection, bag_weight
        init_weights(self)

    def forward(self, x):
        n = x.shape[0]
        h = self.residual(self.stem(x))
        quota = self.instances_per_group
        order = list(range(n))
        random.shuffle(order)
        scores = torch.zeros(n, device=x.device, dtype=torch.float32)
        sub_logits, selected = [], []
        for chunk in np.array_split(np.array(order), self.num_groups):
            index = torch.as_tensor(chunk, device=x.device, dtype=torch.long)
            group = h[index]
            group_scores = self.tier1_attention(group)
            scores[index] = group_scores.float()
            weighted = group * torch.softmax(group_scores, dim=0)[:, None]
            pooled = weighted.sum(dim=0, keepdim=True)
            sub_logits.append(self.tier1_classifier(self.tier1_dropout(pooled)))
            cam = weighted @ self.tier1_classifier.weight.t()
            rank = torch.sort(torch.softmax(cam, dim=1)[:, -1], descending=True).indices
            top, bottom = rank[:quota], rank[-quota:]
            if self.selection == "MaxMinS":
                selected.append(group[torch.cat([top, bottom])])
            elif self.selection == "MaxS":
                selected.append(group[top])
            else:
                selected.append(pooled)
        instances = torch.cat(selected, dim=0)
        feature = attention_pool(self.tier2_attention(instances), instances)
        return {"logits": self.tier2_classifier(feature), "scores": scores, "feature": feature, "sub_logits": torch.cat(sub_logits, dim=0)}

    def loss(self, output, label):
        bag = nn.functional.cross_entropy(output["logits"], label)
        sub = nn.functional.cross_entropy(output["sub_logits"], label.expand(output["sub_logits"].shape[0]))
        return self.bag_weight * bag + (1.0 - self.bag_weight) * sub
