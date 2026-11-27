import numpy as np
from sklearn.metrics import roc_auc_score


def lesion_masks(lesion_fraction, threshold=0.5):
    fraction = np.asarray(lesion_fraction, dtype=np.float64)
    return fraction >= threshold, fraction == 0


def localization_metrics(scores, lesion, background):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    lesion, background = np.asarray(lesion, dtype=bool), np.asarray(background, dtype=bool)
    if not lesion.any() or not background.any():
        return None
    keep = lesion | background
    labels, kept = lesion[keep].astype(int), scores[keep]
    order = np.argsort(-kept, kind="stable")
    num_lesion = int(labels.sum())
    top = max(1, int(round(0.05 * len(labels))))
    return {
        "auroc": float(roc_auc_score(labels, kept)),
        "r_precision": float(labels[order[:num_lesion]].mean()),
        "precision_at_5_percent": float(labels[order[:top]].mean()),
        "lesion_share": float(labels.mean()),
        "num_lesion": num_lesion,
        "num_patches": int(len(labels)),
    }


def mean_localization(per_slide):
    valid = [m for m in per_slide if m is not None]
    keys = ("auroc", "r_precision", "precision_at_5_percent", "lesion_share")
    return {**{k: float(np.mean([m[k] for m in valid])) for k in keys}, "num_slides": len(valid)}
