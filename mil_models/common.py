import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.tanh_branch = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.gate_branch = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.score(self.tanh_branch(h) * self.gate_branch(h))[:, 0]


def attention_pool(scores, h):
    weights = torch.softmax(scores.float(), dim=0).to(h.dtype)
    return (weights[:, None] * h).sum(dim=0, keepdim=True)


def init_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
