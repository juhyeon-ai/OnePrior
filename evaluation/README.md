# Evaluation

Two sets of metrics: how well the attention of a model finds the lesion without training on the task (zero-shot localization), and how well a trained model classifies slides (downstream).

```python
from evaluation import lesion_masks, localization_metrics, mean_localization, classification_metrics
```

## Zero-shot lesion localization (`localization.py`)

A slide has one attention score per patch and, from the pathologist's annotation, the fraction of each patch that is covered by lesion.

```python
lesion, background = lesion_masks(lesion_fraction)
metrics = localization_metrics(scores, lesion, background)
summary = mean_localization([metrics_slide_1, metrics_slide_2])
```

- **Lesion patch**: lesion fraction of at least 0.5. **Background patch**: lesion fraction of exactly 0. Patches in between are left out.
- A slide without a lesion patch or without a background patch gives `None` and is skipped by `mean_localization`.
- Metrics are computed per slide and then averaged over slides.

| Metric | Definition |
|---|---|
| `auroc` | area under the ROC curve of the scores, lesion against background patches |
| `r_precision` | share of lesion patches among the R highest-scored patches, where R is the number of lesion patches of the slide |
| `precision_at_5_percent` | share of lesion patches among the 5 % highest-scored patches |
| `lesion_share` | share of lesion patches among the evaluated patches, the value a random ranking would reach in R-precision |

R-precision is the stricter of the two main metrics: a slide with a small lesion can have a high AUROC while few of its top patches are lesion.

## Downstream classification (`classification.py`)

```python
metrics = classification_metrics(labels, probabilities)
```

`labels` are integer slide labels and `probabilities` is `[num_slides, num_classes]`.

| Metric | Definition |
|---|---|
| `accuracy` | share of correctly classified slides |
| `balanced_accuracy` | mean recall over the classes |
| `macro_f1` | mean F1 score over the classes |
| `auroc` | ROC AUC; one-vs-rest macro average for more than two classes |
| `qwk` | quadratic weighted kappa, for ordinal labels such as ISUP grade |

Which metric is the main one depends on the task: AUROC for binary detection, balanced accuracy for subtyping, QWK for grading.
