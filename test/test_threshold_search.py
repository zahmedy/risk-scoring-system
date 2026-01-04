import numpy as np
from risk_system.evaluate import find_best_threshold

def test_find_best_threshold_fn_cost_pushes_threshold_down():
    # scores: higher means more likely positive (default=1)
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    y_true = np.array([1,   1,   0,   0,   0,   0])

    grid = {"start": 0.05, "stop": 0.95, "step": 0.05}

    low_fn = find_best_threshold(
        y_true=y_true,
        scores=scores,
        objective="min_cost",
        costs={"fp": 1.0, "fn": 1.0},
        grid=grid,
    )["threshold"]

    high_fn = find_best_threshold(
        y_true=y_true,
        scores=scores,
        objective="min_cost",
        costs={"fp": 1.0, "fn": 10.0},
        grid=grid,
    )["threshold"]

    assert high_fn <= low_fn
