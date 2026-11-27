import torch.nn as nn

from .common import GatedAttention, attention_pool, init_weights


class ABMIL(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=512, attn_dim=384, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.attention = GatedAttention(embed_dim, attn_dim, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        init_weights(self)

    def forward(self, x):
        h = self.stem(x)
        scores = self.attention(h)
        feature = attention_pool(scores, h)
        return {"logits": self.classifier(feature), "scores": scores, "feature": feature}
