import numpy as np
import pandas as pd



def monitor(cfg_base: dict, monitor_cfg: dict, artifacts_dir="artifacts") -> dict:
    """
    Loads new data from cfg_base, loads baseline_stats,
    computes drift metrics, writes artifacts/monitor_report.json
    """

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

