# Feature extraction

All patch features in this project were extracted with [TRIDENT](https://github.com/mahmoodlab/TRIDENT): tissue segmentation, patching and patch encoding. Install TRIDENT by following its own instructions; it is not bundled here.

## Settings

| Setting | Value |
|---|---|
| Magnification | 10x |
| Patch size | 256 x 256 pixels |
| Overlap | 0 |
| Output | one `.h5` file per slide with `features` `[num_patches, feature_dim]` and `coords` `[num_patches, 2]` |

The anchor patches in [`../anchors/`](../anchors/) have the same magnification and size, so slide patches and anchors are comparable.

## 1. Slide features

```
bash feature_extraction/extract_features.sh <TRIDENT directory> <slide directory> <output directory> <encoder>
```

This calls TRIDENT's `run_batch_of_slides.py` with the settings above. `<encoder>` is a TRIDENT patch encoder name such as `uni_v1`, `conch_v15` or `virchow2`.

## 2. Anchor features

Encode the 768 anchor patches once per encoder, with the same TRIDENT encoder and its own preprocessing. All 768 patches must be present first; see [`../anchors/`](../anchors/) for the HISTAI and SPIDER patches.

```
python feature_extraction/encode_anchors.py --encoder uni_v1 --out anchor_features_uni_v1.pt
```

The result is a `[768, feature_dim]` tensor. Together with the slide features it gives the anchor coordinates that the prior reads (see the quick start in the main README).

An encoder that is not in TRIDENT works the same way: pass your model and its image transform to `anchors.encode_anchors`.
