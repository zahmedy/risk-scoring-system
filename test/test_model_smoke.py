import pandas as pd
from pathlib import Path

from risk_system.train import train

def test_train_smoke_writes_artifacts(tmp_path):
    df = pd.DataFrame({
        "age": [25, 40, 35, 50, 28, 60, 45, 33],
        "income": [50.0, 80.0, 65.0, 90.0, 55.0, 120.0, 85.0, 70.0],
        "job": ["a", "b", "a", "c", "b", "c", "a", "b"],
        "class": [0, 1, 0, 1, 0, 1, 1, 0],
    })

    csv_path = tmp_path / "risk.csv"
    df.to_csv(csv_path, index=False)

    cfg_base = {
        "dataset": {"path": str(csv_path), "target": "class"},
        "split": {"test_size": 0.25, "random_state": 42, "stratify": True},
    }

    cfg_model = {
        "type": "rf",
        "implementation": "scratch",
        "params": {
            "n_estimators": 20,
            "max_depth": 4,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "seed": 42,
        },
    }

    artifacts_dir = tmp_path / "artifacts"
    metrics = train(cfg_base, cfg_model, artifacts_dir=str(artifacts_dir))

    # artifact existence
    assert (artifacts_dir / "preprocessor.joblib").exists()
    assert (artifacts_dir / "model.joblib").exists()
    assert (artifacts_dir / "schema.json").exists()
    assert (artifacts_dir / "metrics.json").exists()

    # metrics content
    for key in ["accuracy", "confusion_matrix", "roc_auc", "pr_auc", "n_train", "n_test"]:
        assert key in metrics
