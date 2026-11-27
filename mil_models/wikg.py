import torch
import torch.nn as nn

from .common import attention_pool, init_weights


class WiKG(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=512, topk=4, dropout=0.25, chunk=4096):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.head = nn.Linear(embed_dim, embed_dim)
        self.tail = nn.Linear(embed_dim, embed_dim)
        self.sum_proj = nn.Linear(embed_dim, embed_dim)
        self.prod_proj = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.readout_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.LeakyReLU(), nn.Linear(embed_dim // 2, 1))
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.scale, self.topk, self.chunk = embed_dim ** -0.5, topk, chunk
        init_weights(self)

    def nodes(self, x):
        h = self.stem(x)
        h = (h + h.mean(dim=0, keepdim=True)) * 0.5
        heads, tails = self.head(h), self.tail(h)
        out = []
        for start in range(0, h.shape[0], self.chunk):
            head = heads[start: start + self.chunk]
            weight, index = torch.topk((head @ tails.t()) * self.scale, k=self.topk, dim=-1)
            neighbours = tails[index]
            relation = torch.softmax(weight, dim=-1)[..., None] * neighbours
            gate = torch.tanh(head[:, None, :] + relation)
            mix = torch.softmax((gate * neighbours).sum(dim=-1), dim=-1)[..., None]
            aggregated = (mix * neighbours).sum(dim=1)
            out.append(self.activation(self.sum_proj(head + aggregated)) + self.activation(self.prod_proj(head * aggregated)))
        return torch.cat(out, dim=0)

    def forward(self, x):
        h = self.dropout(self.nodes(x))
        scores = self.readout_gate(h)[:, 0]
        feature = self.norm(attention_pool(scores, h))
        return {"logits": self.classifier(feature), "scores": scores, "feature": feature}
