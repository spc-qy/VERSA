import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ======================================================================
# 8. Metric calculation
# ======================================================================
def calc_metrics(y_true, y_prob, y_pred):
    metrics = {}

    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["AUC"] = np.nan

    metrics["ACC"] = accuracy_score(y_true, y_pred)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["CM"] = confusion_matrix(y_true, y_pred)

    return metrics


def format_mean_std(series):
    vals = np.array(series, dtype=np.float64)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return "nan ± nan"
    if len(vals) == 1:
        return f"{vals.mean():.4f} ± 0.0000"

    return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"

# ======================================================================
# 8.1 Threshold search
# ======================================================================
def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= thr).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1
