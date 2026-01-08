import numpy as np
from decision_tree.criteria import gini_gain


def best_split(X: np.ndarray, y: np.ndarray, feature_indices=None):
    n_samples, n_features = X.shape

    best_feature = None
    best_threshold = None
    best_gain = 0.0
    best_left_idx = None
    best_right_idx = None

    if feature_indices == None:
        feature_indices = range(n_features)

    for feature_index in feature_indices:
        values = X[:, feature_index]
        unique_vals = np.unique(values)

        if unique_vals.size < 2:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            # 1) compute left_idx and right_idx using np.where
            left_idx = np.where(values <= threshold)[0]
            right_idx = np.where(values > threshold)[0]

            # 2) skip invalid splits
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            # 3) compute gain (use y, y[left_idx], y[right_idx])
            gain = gini_gain(y, y[left_idx], y[right_idx])

            # 4) update best_* if gain improves
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = float(threshold)
                best_left_idx = left_idx
                best_right_idx = right_idx

    return best_feature, best_threshold, best_gain, best_left_idx, best_right_idx


