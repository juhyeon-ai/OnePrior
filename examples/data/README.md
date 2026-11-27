# Example data

Four test slides of CAMELYON16 with tumour annotation, in the order of their names: `test_001`, `test_002`, `test_008`, `test_016`.

| File | Contents |
|---|---|
| `camelyon16_<slide>_hibou_b.h5` | `features` `[num_patches, 768]` (Hibou-B, extracted with TRIDENT at 10x, 256-pixel patches, stored as float16), `coords` `[num_patches, 2]`, `lesion_fraction` `[num_patches]` (share of each patch covered by the tumour annotation) |
| `anchor_features_hibou_b.pt` | `[768, 768]` Hibou-B features of the anchor patches, the output of `feature_extraction/encode_anchors.py --encoder hibou_b` |

The anchor features are included so that the example runs before the HISTAI and SPIDER anchor patches have been rebuilt.

- CAMELYON16 slides and annotations are released under CC0.
- Hibou-B is released under Apache 2.0.
- The slides were not chosen by result: they are the first four annotated tumour slides of the test set. `test_008` has only two lesion patches and the prior misses both in its top two.
