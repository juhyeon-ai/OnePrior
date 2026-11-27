import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import init_weights


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, dim, heads, gated=False):
        super().__init__()
        self.query = nn.Linear(query_dim, dim)
        self.key = nn.Linear(key_dim, dim)
        self.value = nn.Linear(key_dim, dim)
        self.mha = nn.MultiheadAttention(dim, heads)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(query_dim, dim), nn.SiLU()) if gated else None

    def forward(self, queries, keys, need_weights=False):
        q = self.query(queries).transpose(0, 1)
        k, v = self.key(keys).transpose(0, 1), self.value(keys).transpose(0, 1)
        attended, weights = self.mha(q, k, v, need_weights=need_weights, average_attn_weights=True)
        h = self.norm_attn((q + attended).transpose(0, 1))
        h = self.norm_out(h + F.relu(self.out(h)))
        if self.gate is not None:
            h = h * self.gate(queries)
        return h, weights


class LowRankBlock(nn.Module):
    def __init__(self, in_dim, dim, heads, rank):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(1, rank, dim))
        nn.init.xavier_uniform_(self.latent)
        self.down = CrossAttention(dim, in_dim, dim, heads, gated=True)
        self.up = CrossAttention(in_dim, dim, dim, heads, gated=True)

    def forward(self, x):
        low, _ = self.down(self.latent, x)
        return self.up(x, low)[0]


class NonLocalPooling(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.token)
        self.attn = CrossAttention(dim, dim, dim, heads)

    def forward(self, x):
        pooled, weights = self.attn(self.token, x, need_weights=True)
        return pooled, weights[0].sum(dim=0)


class ILRA(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=256, num_layers=2, num_heads=8, rank=64):
        super().__init__()
        self.blocks = nn.ModuleList([LowRankBlock(in_dim if i == 0 else embed_dim, embed_dim, num_heads, rank) for i in range(num_layers)])
        self.pool = NonLocalPooling(embed_dim, num_heads)
        self.classifier = nn.Linear(embed_dim, num_classes)
        init_weights(self)

    def forward(self, x):
        h = x[None]
        for block in self.blocks:
            h = block(h)
        pooled, attention = self.pool(h)
        feature = pooled.reshape(1, -1)
        return {"logits": self.classifier(feature), "scores": torch.log(attention.float() + 1e-12), "feature": feature}
