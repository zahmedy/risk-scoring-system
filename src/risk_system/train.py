from risk_system.preprocess import make_preprocessor, infer_feature_types
from risk_system.data import load_csv, split_X_y, train_test_split_df, infer_schema

import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

def _to_dense(Xt):
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


def train(cfg_base: dict, cfg_model: dict, artifacts_dir: str = "artifacts") -> dict:
    # Load & split 
    df = load_csv(cfg_base)
    X, y = split_X_y(df, target=cfg_base["dataset"]["target"])
    X_train, X_test, y_train, y_test = train_test_split_df(X, y, cfg_base)

    # Features preprocessing
    num_features, cat_features = infer_feature_types(X_train)
    preprocessor = make_preprocessor(num_features, cat_features, model_type=cfg_model["type"])
    X_train_t = _to_dense(preprocessor.fit_transform(X_train))
    X_test_t = _to_dense(preprocessor.transform(X_test))
    
    # Build model, fit and predict 
    model = _build_model(cfg_model)
    model.fit(X_train_t, y_train.to_numpy())
    y_pred = model.predict(X_test_t)
    
    # Accuracy and Confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    metrics = {"accuracy": accuracy, 
            "confusion_matrix": matrix.tolist(), 
            "n_train": len(y_train), 
            "n_test": len(y_test)}

    # Save artifacts
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, f"{artifacts_dir}/preprocessor.joblib")
    joblib.dump(model, f"{artifacts_dir}/model.joblib")
    schema = infer_schema(
        X_train.join(y_train), 
        target=cfg_base["dataset"]["target"]
    )

    with open(f"{artifacts_dir}/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    with open(f"{artifacts_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



