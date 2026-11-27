# Downstream transfer

`akd.py` transfers the prior to a downstream MIL model by attention knowledge distillation (AKD).

The downstream model is trained on the raw patch features of the task as usual. The prior stays frozen. It reads the same slide through its anchor coordinates and gives an attention distribution over the patches, and the downstream model is asked to attend to the same patches early in training.

## Loss

For one slide,

```
loss = task loss + beta(epoch) * KL(prior attention || model attention)
```

- Both attention distributions are a softmax of the per-patch scores over the patches of the slide.
- `beta(epoch) = weight * max(0, 1 - (epoch - 1) / (epochs // 2))`. The guidance starts at `weight` and falls linearly to zero over the first half of training; the second half is trained on the task loss alone.
- The task loss is cross-entropy. For `dtfd` it is the model's own two-tier loss.

## Defaults

| Setting | Value |
|---|---|
| `weight` | 0.3 |
| Epochs | 20 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-5 |
| Schedule | cosine to 5e-6 |
| Batch | one slide |

## Usage

```python
import torch
from mil_models import build_model, load_prior
from downstream_transfer.akd import fit

prior = build_model("abmil", in_dim=768, num_classes=2)
load_prior(prior, "weights/abmil_scout.pt")
anchor_features = torch.load("anchor_features_uni_v1.pt")

model = build_model("transmil", in_dim=1024, num_classes=2)
history = fit(model, prior, anchor_features, bags, labels, device="cuda")
```

| Name | Meaning |
|---|---|
| `bags` | list of `[num_patches, feature_dim]` tensors, one per training slide |
| `labels` | list of integer slide labels |
| `anchor_features` | anchor features from the same patch encoder as `bags` |
| `history` | mean training loss per epoch |

Any of the five architectures can be the downstream model. `akd.py` also exposes the pieces (`akd_weight`, `akd_loss`, `train_epoch`) for use inside another training loop.
