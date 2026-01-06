from risk_system.preprocess import make_preprocessor, infer_feature_types
from risk_system.data import load_csv, split_X_y, train_test_split_df, infer_schema
from risk_system.monitor import (
    make_numeric_bins,
    histogram_proportions,
    categorical_proportions,
)
from risk_system.evaluate import predict_scores

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             average_precision_score)
from pathlib import Path


def _to_dense(Xt) -> np.ndarray:
    if hasattr(Xt, "toarray"):
        return Xt.toarray()
    return Xt


def _build_model(cfg_model: dict):
    if cfg_model["type"] != "rf":
        raise ValueError("Only running Random Forest model for now.")
    if cfg_model["implementation"] == "scratch":
        from decision_tree.random_forest import RandomForestClassifier
        return RandomForestClassifier(**cfg_model["params"])
    else:
        raise ValueError(f"Unsupported implementation: {cfg_model['implementation']}")

############### TRAIN #################
def train(cfg_base: dict, cfg_model: dict, cfg_monitor: dict, artifacts_dir: str = "artifacts") -> dict:
    # Load & split 
    df = load_csv(cfg_base)
    X, y = split_X_y(df, target=cfg_base["dataset"]["target"])
    X_train, X_test, y_train, y_test = train_test_split_df(X, y, cfg_base)

    # Encoding
    y_train = (y_train == "bad").astype(int).to_numpy()
    y_test  = (y_test == "bad").astype(int).to_numpy()

    # Features preprocessing
    num_features, cat_features = infer_feature_types(X_train)
    preprocessor = make_preprocessor(num_features, cat_features, model_type=cfg_model["type"])
    X_train_t = _to_dense(preprocessor.fit_transform(X_train))
    X_test_t = _to_dense(preprocessor.transform(X_test))
    
    # Build model, fit and predict 
    model = _build_model(cfg_model)
    model.fit(X_train_t, y_train)

    # get train scores for baseline score drift
    scores_train = predict_scores(model, X_train_t)

    baseline_stats = build_baseline_stats(
        X_train=X_train,
        scores_train=scores_train,
        numeric_features=num_features,
        categorical_features=cat_features,
        monitor_cfg=cfg_monitor,
    )

    with open(Path(artifacts_dir) / "baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    y_pred = model.predict(X_test_t)
    
    # Accuracy and Confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    roc_auc = None
    pr_auc = None
    proba_summary = None

    if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
        proba = model.predict_proba(X_test_t)
        y_score = proba[:, 1]
        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)
        proba_summary = {
            "min": float(y_score.min()),
            "mean": float(y_score.mean()),
            "max": float(y_score.max())
        }
    
    pos_rate_test = float(np.mean(y_test))
    pos_rate_train = float(np.mean(y_train))

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),              # optional
        "confusion_matrix": matrix.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pos_rate_train": pos_rate_train,
        "pos_rate_test": pos_rate_test,
        "proba_summary": proba_summary,
    }

    # Save artifacts
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, f"{artifacts_dir}/preprocessor.joblib")
    joblib.dump(model, f"{artifacts_dir}/model.joblib")
    schema = infer_schema(
        X_train, 
        target=cfg_base["dataset"]["target"]
    )

    with open(f"{artifacts_dir}/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    with open(f"{artifacts_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
############### END TRAIN #################

def build_baseline_stats(
    X_train,
    scores_train,
    numeric_features,
    categorical_features,
    monitor_cfg: dict,
) -> dict:
    n_bins_num = int(monitor_cfg["numeric"]["n_bins"])
    top_k = int(monitor_cfg["categorical"]["top_k"])
    n_bins_score = int(monitor_cfg["score"]["n_bins"])

    baseline = {"numeric": {}, "categorical": {}, "score": {}}

    # --- numeric features ---
    for col in numeric_features:
        x = X_train[col].to_numpy()
        edges = make_numeric_bins(x, n_bins=n_bins_num)
        props = histogram_proportions(x, edges)
        baseline["numeric"][col] = {
            "bin_edges": edges.tolist(),
            "proportions": props.tolist(),
        }

    # --- categorical features ---
    for col in categorical_features:
        props_map = categorical_proportions(X_train[col], top_k=top_k)
        baseline["categorical"][col] = {"proportions": props_map}

    # --- score distribution ---
    # fixed edges is fine for scores (0..1), but we’ll use quantile edges for consistency
    edges_s = make_numeric_bins(scores_train, n_bins=n_bins_score)
    props_s = histogram_proportions(scores_train, edges_s)
    baseline["score"] = {
        "bin_edges": edges_s.tolist(),
        "proportions": props_s.tolist(),
    }

    baseline["n_train_rows"] = int(len(X_train))

    return baseline

