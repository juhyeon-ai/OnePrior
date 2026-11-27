import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    binary = probabilities.shape[1] == 2
    try:
        auroc = float(roc_auc_score(labels, probabilities[:, 1]) if binary
                      else roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = None
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "auroc": auroc,
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
    }
