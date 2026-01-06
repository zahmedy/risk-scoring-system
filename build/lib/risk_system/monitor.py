from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from risk_system.data import load_csv, split_X_y
from risk_system.preprocess import infer_feature_types
from risk_system.evaluate import predict_scores  # reuse this


def status_from_psi(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return "CRIT"
    if value >= warn:
        return "WARN"
    return "OK"

def make_numeric_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    # quantile bins
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]  # ignore NaN for bin-building 

    # quantiles from 0 to 1 
    qs = np.linspace(0.0, 1.0, n_bins + 1)

    edges = np.quantile(x, qs)

    # IMPORTANT: edges must be strictly increasing
    edges = np.unique(edges)

    return edges


def histogram_proportions(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # returns proportions that sum to 1
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    counts , _ = np.histogram(x, bins=bin_edges)

    total = counts.sum()
    if total == 0:
        # edge case: no data
        return np.zeros(len(bin_edges) - 1)
    
    proportions = counts / total
    return proportions

def psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    # p = Baseline proporties 
    # q = current proportions 
    # PSI: i∑​(pi​−qi​)⋅ln(qi​pi​​)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # safety: normalize
    p = p / p.sum()
    q = q / q.sum()

    # avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    
    return float(np.sum((p - q) * np.log(p / q)))

def categorical_proportions(s: pd.Series, top_k: int = 20) -> dict[str, float]:
    # map categories to proportions, others -> "__OTHER__"
    s = s.astype("object").fillna("__MISSING__")

    vc = s.value_counts(dropna=False)
    total = float(vc.sum())

    top = vc.head(top_k)
    other_count = vc.iloc[top_k:].sum()

    props = {str(k): float(v / total) for k, v in top.items()}
    if other_count > 0:
        props["__OTHER__"] = float(other_count / total)

    return props

def categorical_psi(baseline_props: dict[str,float], 
                    current_props: dict[str,float],
                    eps: float = 1e-6
) -> float:
    # align keys + fill missing with eps, then psi
    keys = sorted(set(baseline_props) | set(current_props))

    p = np.array([baseline_props.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([current_props.get(k, 0.0) for k in keys], dtype=float)

    return psi(p, q, eps=eps)


# ---------- Monitor job ----------

def monitor(cfg_base: dict, monitor_cfg: dict, artifacts_dir: str = "artifacts") -> dict:
    # 1) Load current data (labels not required)
    df = load_csv(cfg_base)
    target = cfg_base["dataset"]["target"]
    X, _ = split_X_y(df, target=target)

    # 2) Load baseline stats
    baseline_path = monitor_cfg["baseline_stats_path"]
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # 3) Load model + preprocessor for score drift
    artifacts_dir_p = Path(artifacts_dir)
    preprocessor = joblib.load(artifacts_dir_p / "preprocessor.joblib")
    model = joblib.load(artifacts_dir_p / "model.joblib")

    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    scores = predict_scores(model, Xt)

    # 4) Feature typing from current X (should match training schema)
    numeric_features, cat_features = infer_feature_types(X)

    # thresholds
    warn_n = float(monitor_cfg["alerts"]["psi_numeric_warn"])
    crit_n = float(monitor_cfg["alerts"]["psi_numeric_crit"])
    warn_c = float(monitor_cfg["alerts"]["psi_categorical_warn"])
    crit_c = float(monitor_cfg["alerts"]["psi_categorical_crit"])
    warn_s = float(monitor_cfg["alerts"]["psi_score_warn"])
    crit_s = float(monitor_cfg["alerts"]["psi_score_crit"])

    # 5) Compute numeric drift
    numeric_report: dict[str, dict] = {}
    for col in numeric_features:
        b = baseline.get("numeric", {}).get(col)
        if b is None:
            continue

        edges = np.array(b["bin_edges"], dtype=float)
        p = np.array(b["proportions"], dtype=float)
        q = histogram_proportions(X[col].to_numpy(), edges)

        val = psi(p, q)
        numeric_report[col] = {
            "psi": float(val),
            "status": status_from_psi(float(val), warn=warn_n, crit=crit_n),
        }

    # 6) Compute categorical drift
    categorical_report: dict[str, dict] = {}
    top_k = int(monitor_cfg["categorical"]["top_k"])

    for col in cat_features:
        b = baseline.get("categorical", {}).get(col)
        if b is None:
            continue

        p_map = b["proportions"]
        q_map = categorical_proportions(X[col], top_k=top_k)

        val = categorical_psi(p_map, q_map)
        categorical_report[col] = {
            "psi": float(val),
            "status": status_from_psi(float(val), warn=warn_c, crit=crit_c),
        }

    # 7) Compute score drift
    score_psi = None
    if "score" in baseline:
        score_edges = np.array(baseline["score"]["bin_edges"], dtype=float)
        p_score = np.array(baseline["score"]["proportions"], dtype=float)
        q_score = histogram_proportions(scores, score_edges)
        score_psi = psi(p_score, q_score)

    score_status = None
    if score_psi is not None:
        score_status = status_from_psi(float(score_psi), warn=warn_s, crit=crit_s)

    # 8) Overall status (simple rule: CRIT > WARN > OK)
    statuses = [d["status"] for d in numeric_report.values()] + [d["status"] for d in categorical_report.values()]
    if score_status is not None:
        statuses.append(score_status)

    overall = "OK"
    if "CRIT" in statuses:
        overall = "CRIT"
    elif "WARN" in statuses:
        overall = "WARN"

    report = {
        "summary": {
            "overall_status": overall,
            "n_rows": int(len(X)),
            "avg_score": float(np.mean(scores)) if len(scores) else None,
            "score_psi": None if score_psi is None else float(score_psi),
            "score_status": score_status,
        },
        "numeric": numeric_report,
        "categorical": categorical_report,
    }

    # 9) Write report
    artifacts_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir_p / "monitor_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report