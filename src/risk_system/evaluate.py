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

def evaluate(cfg_base, artifacts_dir="artifacts", 
             threshold=0.5, 
             policy: dict | None = None
) -> dict:
    
    df = load_csv(cfg_base)
    X,y = split_X_y(df, target=cfg_base["dataset"]["target"])

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    model = joblib.load(f"{artifacts_dir}/model.joblib")
    preprocessor = joblib.load(artifacts_dir+"/preprocessor.joblib")

    Xt = _to_dense(preprocessor.transform(X))
    scores = predict_scores(model, Xt)
    scores = np.asarray(scores, dtype=float)

    threshold_search = None

    if policy is not None:
        mode = policy.get("mode", "fixed")
        if mode == "fixed":
            threshold = float(policy["threshold"])
        elif mode == "search":
            threshold_search = find_best_threshold(
                y_true=y,
                scores=scores,
                objective=policy.get("objective", "min_cost"),
                costs=policy.get("costs"),
                grid=policy.get("grid"),
            )
            threshold = float(threshold_search["threshold"])
        else:
            raise ValueError(f"Unknown policy mode: {mode}")


    y_pred = apply_threshold(scores, threshold)
    metrics = compute_metrics(y, y_pred, scores)
    metrics["threshold_used"] = float(threshold)
    metrics["threshold_search"] = threshold_search


    with open(f"{artifacts_dir}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def find_best_threshold(
    y_true,
    scores: np.ndarray,
    objective: str = "min_cost",
    costs: dict | None = None,
    grid: dict | None = None,
) -> dict:
    if grid is None:
        raise ValueError("grid is required")

    start = float(grid.get("start", 0.01))
    stop = float(grid.get("stop", 0.99))
    step = float(grid.get("step", 0.01))
    thresholds = np.arange(start, stop + 1e-12, step)

    if objective == "min_cost":
        if costs is None:
            costs = {"fp": 1.0, "fn": 1.0}
        fp_cost = float(costs["fp"])
        fn_cost = float(costs["fn"])

        best_value = float("inf")
        best_threshold = float(thresholds[0])

        for t in thresholds:
            y_pred = apply_threshold(scores, threshold=float(t))
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = fp_cost * fp + fn_cost * fn
            if cost < best_value:
                best_value = float(cost)
                best_threshold = float(t)

    elif objective == "max_f1":
        best_value = float("-inf")
        best_threshold = float(thresholds[0])

        for t in thresholds:
            y_pred = apply_threshold(scores, threshold=float(t))
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_value:
                best_value = float(f1)
                best_threshold = float(t)

    else:
        raise ValueError(f"Unknown objective: {objective}")

    return {
        "threshold": best_threshold,
        "objective": objective,
        "value": best_value,
    }

