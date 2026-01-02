import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score, 
    precision_recall_curve
)


def predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1].astype(float)
    
    pred = model.predict(X)
    return pred.astype(float)

def apply_threshold(scores, threshold) -> np.ndarray:
    return (scores >= threshold).astype(int)

def compute_metrics(y_true, y_pred, scores=None) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    roc_auc = None
    pr_auc = None
    if scores is not None and len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_score=scores)
        pr_auc = average_precision_score(y_true, y_score=scores)
        pr_curve = precision_recall_curve(y_true, scores)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc)
    }
    return metrics
