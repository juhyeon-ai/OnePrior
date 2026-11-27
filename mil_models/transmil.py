import math

import torch
import torch.nn as nn
from nystrom_attention import NystromAttention

from .common import init_weights


class NystromLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim=dim, dim_head=dim // heads, heads=heads, num_landmarks=dim // 2,
                                     pinv_iterations=6, residual=True, dropout=0.1)

    def forward(self, tokens):
        return tokens + self.attn(self.norm(tokens))

    def cls_attention(self, tokens):
        q, k, _ = self.attn.to_qkv(self.norm(tokens)).chunk(3, dim=-1)
        heads = self.attn.heads
        q = q[:, :1].reshape(1, 1, heads, -1).transpose(1, 2)
        k = k.reshape(1, k.shape[1], heads, -1).transpose(1, 2)
        return torch.softmax(((q * self.attn.scale) @ k.transpose(-1, -2)).float(), dim=-1).mean(dim=1)[0, 0]


class PyramidPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, tokens, side):
        cls_token, patches = tokens[:, :1], tokens[:, 1:]
        grid = patches.transpose(1, 2).reshape(1, -1, side, side)
        grid = self.conv7(grid) + grid + self.conv5(grid) + self.conv3(grid)
        return torch.cat((cls_token, grid.flatten(2).transpose(1, 2)), dim=1)


class TransMIL(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=512, num_layers=2, num_heads=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([NystromLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.position = PyramidPositionEncoding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        init_weights(self)

    def forward(self, x):
        n = x.shape[0]
        tokens = self.stem(x[None])
        side = int(math.ceil(math.sqrt(n)))
        tokens = torch.cat([tokens, tokens[:, : side * side - n]], dim=1)
        tokens = torch.cat((self.cls_token.expand(1, -1, -1), tokens), dim=1)
        scores = None
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                scores = torch.log(layer.cls_attention(tokens)[1: n + 1] + 1e-12)
            tokens = layer(tokens)
            if i == 0:
                tokens = self.position(tokens, side)
        feature = self.norm(tokens)[:, 0]
        return {"logits": self.classifier(feature), "scores": scores, "feature": feature}
