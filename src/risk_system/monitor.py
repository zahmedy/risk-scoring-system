import numpy as np



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


# def histogram_proportions(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # returns proportions that sum to 1

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

# def categorical_proportions(series: pd.Series, top_k: int) -> dict[str, float]:
    # map categories to proportions, others -> "__OTHER__"

# def categorical_psi(baseline: dict[str,float], current: dict[str,float]) -> float:
    # align keys + fill missing with eps, then psi
