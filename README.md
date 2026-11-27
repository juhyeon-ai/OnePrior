<h1 align="center">One Prior, Every Encoder</h1>

<p align="center">
  <b>Distilling Where to Look into Histopathology MIL</b><br>
  Juhyeon Kim &nbsp;·&nbsp; Chanjae Song &nbsp;·&nbsp; Mun Yong Yi<br>
  <i>KAIST</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
</p>

**Scout** is a pretrained attention prior for whole-slide image (WSI) multiple instance learning (MIL). It tells a MIL aggregator *where to look* in a slide, and one frozen Scout works with any patch encoder, including encoders it never saw during pretraining.

- **One prior, any encoder.** Slides are read in anchor coordinates, a 768-dimensional space shared by all patch encoders. Supporting a new encoder takes one pass over 768 anchor patches.
- **Zero-shot localization.** Without any task training, Scout's attention ranks lesion patches in a slide.
- **Better downstream MIL.** Scout supervises the attention of a task-specific aggregator through attention knowledge distillation (AKD). Scout is not used at inference.

## Highlights

- **Strong generalization.** Improves downstream performance in 113 of 120 benchmark–encoder pairs, with comparable gains on seen (+3.1 points) and unseen (+2.9 points) encoders.
- **Data efficiency.** Improves performance by approximately 7 points with only four labeled slides per class.
- **Improved interpretability.** Produces attention maps better aligned with annotated lesions, without requiring downstream lesion annotations for training.
- **Minimal overhead.** Adds only 5–10% to downstream training time, with zero additional inference cost.

> **Status.** This repository is being filled step by step. See [What is available](#what-is-available) and the [Roadmap](#roadmap).

## Contents

- [Highlights](#highlights)
- [What is available](#what-is-available)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Use your own slides](#use-your-own-slides)
- [Pretrained weights](#pretrained-weights)
- [Anchor coordinates](#anchor-coordinates)
- [Baseline MIL architectures](#baseline-mil-architectures)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

## What is available

| Folder | Contents |
|---|---|
| [`weights/`](weights/) | Pretrained Scout for ABMIL |
| [`anchors/`](anchors/) | Anchor patches (TCGA included; HISTAI and SPIDER rebuilt from the originals), their sources, and the anchor-coordinate code |
| [`feature_extraction/`](feature_extraction/) | Patch features with TRIDENT, anchor features |
| [`downstream_transfer/`](downstream_transfer/) | Attention knowledge distillation (AKD) to a downstream model |
| [`mil_models/`](mil_models/) | Baseline MIL architectures: ABMIL, DTFD-MIL, TransMIL, WiKG, ILRA |
| [`evaluation/`](evaluation/) | Zero-shot localization metrics (AUROC, R-precision) and downstream metrics (accuracy, balanced accuracy, QWK) |
| [`examples/`](examples/) | Quick start on real slides |

## Installation

```bash
git clone https://github.com/juhyeon-ai/OnePrior.git
cd OnePrior
pip install -e .
```

Feature extraction additionally needs [TRIDENT](https://github.com/mahmoodlab/TRIDENT).

## Quick start

Run Scout on real slides and measure how well its attention finds the tumor, with no training.

```bash
python examples/quickstart.py
```

The repository ships four CAMELYON16 test slides as Hibou-B features with their tumor annotations ([`examples/data/`](examples/data/)). The example turns the features into anchor coordinates, runs the ABMIL Scout, saves the attention of every slide, and evaluates it against the annotation.

```text
slide                            patches  scored  lesion   AUROC  R-prec  random
camelyon16_test_001_hibou_b         4483    4413      90   1.000   0.989   0.020
camelyon16_test_002_hibou_b         3900    3857      33   0.999   0.879   0.009
camelyon16_test_008_hibou_b         6816    6805       2   1.000   0.000   0.000
camelyon16_test_016_hibou_b         6295    6091     645   1.000   0.992   0.106
mean                                                       1.000   0.715   0.034
```

| Column | Meaning |
|---|---|
| `patches` | Number of patches in the slide |
| `scored` | Patches used for the metrics: tumor patches and patches with no tumor. Patches partly covered by tumor (less than half) are left out |
| `lesion` | Number of tumor patches (at least half covered by the annotation) |
| `AUROC` | Tumor against non-tumor patches, ranked by attention |
| `R-prec` | Share of tumor patches among the R most attended patches, with R the number of tumor patches |
| `random` | Expected R-precision of a random ranking, `lesion / scored` |

The metrics are defined in [`evaluation/`](evaluation/).

## Use your own slides

1. **Extract patch features** with [TRIDENT](https://github.com/mahmoodlab/TRIDENT) at 10x magnification and 256-pixel patches ([`feature_extraction/`](feature_extraction/)).
2. **Complete the anchor patches** once ([`anchors/`](anchors/)). The HISTAI and SPIDER patches are rebuilt from your copy of those datasets.
3. **Encode the anchors** with the same encoder, then run Scout.

```bash
python feature_extraction/encode_anchors.py --encoder uni_v1 --out anchor_features_uni_v1.pt
python examples/quickstart.py --features slide.h5 --anchor-features anchor_features_uni_v1.pt
```

The attention is saved as one value per patch, in the order of the feature file. Without an annotation, the example reports the most attended patch instead of the metrics.

## Pretrained weights

| File | Architecture | Size |
|---|---|---|
| `weights/abmil_scout.pt` | `abmil` | 3 MB |

```python
from mil_models import build_model, load_prior

model = build_model("abmil", in_dim=768, num_classes=2)
load_prior(model, "weights/abmil_scout.pt")
```

`load_prior` loads every layer except the classifier.

- **The input is not a raw patch feature.** Scout reads 768-dimensional anchor coordinates (next section).
- **The classifier is not included.** It is created for the downstream task and trained there.
- **One Scout for every aggregator.** The ABMIL Scout supervises any downstream aggregator through AKD (ABMIL, TransMIL, DTFD-MIL, WiKG, ILRA); the downstream architecture does not have to match Scout's. See [`downstream_transfer/`](downstream_transfer/).

## Anchor coordinates

Scout reads every slide in a coordinate system that is the same for all patch encoders. Dimension *k* of a patch's coordinate is the cosine similarity between its feature and the feature of anchor patch *k*, both computed with the same encoder.

The 768 anchor patches are described in [`anchors/`](anchors/):

| Source | Patches | In this repository |
|---|---|---|
| TCGA | 256 | Included |
| HISTAI | 256 | Rebuilt from the original dataset, verified by checksum |
| SPIDER | 256 | Rebuilt from the original dataset, verified by checksum |

To use Scout with a patch encoder, encode the anchor patches once with that encoder and its own image transform, then turn the slide features into coordinates.

```python
import torch
from anchors import encode_anchors, anchor_coordinates
from mil_models import build_model, load_prior

anchor_features = encode_anchors(encoder, transform, device="cuda")
torch.save(anchor_features, "anchor_features.pt")

coordinates = anchor_coordinates(slide_features, anchor_features)

model = build_model("abmil", in_dim=768, num_classes=2).eval()
load_prior(model, "weights/abmil_scout.pt")
attention = torch.softmax(model(coordinates)["scores"], dim=0)
```

| Name | Meaning |
|---|---|
| `encoder`, `transform` | Your patch encoder (images to features) and its image preprocessing |
| `slide_features` | `[num_patches, feature_dim]` features of one slide from the same encoder |
| `coordinates` | `[num_patches, 768]` input of Scout |
| `attention` | `[num_patches]` zero-shot attention of Scout over the slide |

The slide patches should match the anchors: 256 x 256 pixels at 10x magnification. We checked that the anchor images reproduce the anchor features used in pretraining (cosine similarity 1.0000 for ResNet-50, UNI, and CONCH v1.5).

## Baseline MIL architectures

Five aggregators, each in one file under [`mil_models/`](mil_models/).

| Name | Class | Paper | Per-patch score |
|---|---|---|---|
| `abmil` | `ABMIL` | Ilse et al., ICML 2018 | Gated attention logit |
| `dtfd` | `DTFDMIL` | Zhang et al., CVPR 2022 | Tier-1 gated attention logit inside the pseudo-bag |
| `transmil` | `TransMIL` | Shao et al., NeurIPS 2021 | Log attention of the class token over the patches in the last layer |
| `wikg` | `WiKG` | Li et al., CVPR 2024 | Gate logit of the attention readout |
| `ilra` | `ILRA` | Xiang and Zhang, ICLR 2023 | Log attention of the global token in the non-local pooling |

All models take the patch features of one slide and return a dictionary.

```python
import torch
from mil_models import build_model

model = build_model("transmil", in_dim=1024, num_classes=2)
out = model(torch.randn(3000, 1024))
```

| Input / output | Shape | Meaning |
|---|---|---|
| input | `[num_patches, feature_dim]` | Patch features of one slide |
| `out["logits"]` | `[1, num_classes]` | Slide prediction |
| `out["scores"]` | `[num_patches]` | Per-patch score, higher = more attended |
| `out["feature"]` | `[1, embed_dim]` | Slide feature |

`DTFDMIL` also returns `out["sub_logits"]` (one prediction per pseudo-bag) and provides `model.loss(out, label)` for its two-tier objective.

<details>
<summary><b>Default hyperparameters</b></summary>

<br>

The defaults follow the base configurations of [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab), which the experiments used. The code here is our own implementation of the same architectures; each one was checked to give the same outputs as the MIL-Lab model when loaded with the same weights and settings.

| Name | Defaults |
|---|---|
| `abmil` | Embedding 512, attention 384, dropout 0.25 |
| `dtfd` | Embedding 512, attention 128, 4 residual blocks, dropout 0.25, 8 pseudo-bags, `MaxMinS` selection, 1 instance per pseudo-bag, bag weight 0.7 |
| `transmil` | Embedding 512, 2 Nystrom attention layers, 4 heads |
| `wikg` | Embedding 512, 4 neighbors, bi-interaction aggregation, attention readout, dropout 0.25 |
| `ilra` | Embedding 256, 2 low-rank blocks, 8 heads, rank 64 |

</details>

## Roadmap

- [x] Pretrained Scout weights (ABMIL)
- [x] Anchor patches and anchor-coordinate code
- [x] Baseline MIL architectures
- [x] Zero-shot localization quick start
- [ ] AKD training examples for TransMIL, DTFD-MIL, WiKG, and ILRA using the same pretrained Scout
- [ ] Scout variants pretrained with other aggregator architectures (TransMIL, DTFD-MIL, WiKG, ILRA)
- [ ] Full downstream training recipe and benchmark scripts

## Citation

```bibtex
@article{kim2026oneprior,
  title   = {One Prior, Every Encoder: Distilling Where to Look into Histopathology MIL},
  author  = {Kim, Juhyeon and Song, Chanjae and Yi, Mun Yong},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code, weights, and the TCGA anchor patches are released under [CC BY-NC 4.0](LICENSE) for non-commercial research use.

- Scout was pretrained on TCGA and HISTAI slides, and the anchors come from TCGA, HISTAI, and SPIDER.
- HISTAI and SPIDER are CC BY-NC 4.0 and are obtained from their authors; their patches are not redistributed here. See [`anchors/README.md`](anchors/README.md).
- The example slides are CAMELYON16 (CC0) features from Hibou-B (Apache 2.0).

## Contact

Questions and requests: please open an issue.
