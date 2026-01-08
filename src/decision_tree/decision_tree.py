import numpy as np
from decision_tree.node import Node
from decision_tree.split import best_split
from decision_tree.criteria import gini


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, seed=42, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.max_features = max_features
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.n_classes_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes_ = int(np.max(y)) + 1
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _majority_class(self, y: np.ndarray) -> int:
        return int(np.bincount(y).argmax())

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # 1) stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or gini(y) == 0.0:
            counts = np.bincount(y, minlength=self.n_classes_)
            pred = int(counts.argmax())
            return Node(is_leaf=True, prediction=pred, class_counts=counts)
        
        n_features = X.shape[1]

        feature_indices = None
        if self.max_features is None:
            feature_indices = None 
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            feature_indices = self.rng.choice(n_features, size=k, replace=False)
        elif isinstance(self.max_features, int):
            k = min(n_features, max(1, self.max_features))
            feature_indices = self.rng.choice(n_features, size=k, replace=False)
        else:
            raise ValueError("max_features must be None, 'sqrt', or int")


        # 2) find best split
        feature, threshold, gain, left_idx, right_idx = best_split(X, y, feature_indices=feature_indices)

        # 3) if no useful split
        if feature is None or gain <= 1e-12:
            counts = np.bincount(y, minlength=self.n_classes_)
            pred = int(counts.argmax())
            return Node(is_leaf=True, prediction=pred, class_counts=counts)

        # 4) split data and recurse
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        # 5) return internal node
        return Node(
            is_leaf=False,
            feature_index=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf:
            return node.prediction

        # decide direction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
        
    def predict_proba(self, X: np.ndarray):
        return np.vstack([self._predict_proba_one(x, self.root) for x in X])

    def _predict_proba_one(self, x: np.ndarray, node: Node):
        if node.is_leaf:
            proba = node.class_counts / node.class_counts.sum()
            return proba

        # decide direction
        if x[node.feature_index] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        else:
            return self._predict_proba_one(x, node.right)
        