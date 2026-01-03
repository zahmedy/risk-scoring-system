import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)

from risk_system.data import load_csv, split_X_y 
from risk_system.train import _to_dense


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

    y_arr = np.asarray(y_true)
    if scores is not None and len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_arr, scores)
        pr_auc = average_precision_score(y_arr, scores)

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

def evaluate(cfg_base, artifacts_dir="artifacts", threshold=0.5) -> dict:
    df = load_csv(cfg_base)
    X,y = split_X_y(df, target=cfg_base["dataset"]["target"])

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    model = joblib.load(f"{artifacts_dir}/model.joblib")
    preprocessor = joblib.load(artifacts_dir+"/preprocessor.joblib")

    Xt = _to_dense(preprocessor.transform(X))
    scores = predict_scores(model, Xt)
    scores = np.asarray(scores, dtype=float)
    y_pred = apply_threshold(scores, threshold)
    
    metrics = compute_metrics(y, y_pred ,scores)

    with open(f"{artifacts_dir}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics