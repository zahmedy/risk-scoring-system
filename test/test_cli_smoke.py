import subprocess
import sys
from pathlib import Path
import yaml
import pandas as pd

def test_cli_train_and_evaluate_smoke(tmp_path):
    # 1) write csv
    df = pd.DataFrame({
        "age": [25, 40, 35, 50, 28, 60, 45, 33],
        "income": [50.0, 80.0, 65.0, 90.0, 55.0, 120.0, 85.0, 70.0],
        "job": ["a", "b", "a", "c", "b", "c", "a", "b"],
        "class": [0, 1, 0, 1, 0, 1, 1, 0],
    })
    csv_path = tmp_path / "risk.csv"
    df.to_csv(csv_path, index=False)

    # 2) write configs
    base = {
        "dataset": {"path": str(csv_path), "target": "class"},
        "split": {"test_size": 0.25, "random_state": 42, "stratify": True},
    }
    model = {
        "type": "rf",
        "implementation": "scratch",
        "params": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "max_features": "sqrt", "seed": 42},
    }
    policy = {"threshold": 0.5}

    base_path = tmp_path / "base.yaml"
    model_path = tmp_path / "model.yaml"
    policy_path = tmp_path / "policy.yaml"
    base_path.write_text(yaml.safe_dump(base))
    model_path.write_text(yaml.safe_dump(model))
    policy_path.write_text(yaml.safe_dump(policy))

    artifacts_dir = tmp_path / "artifacts"

    # 3) run train
    subprocess.run(
        [sys.executable, "-m", "risk_system.cli", "train", "--base", str(base_path), "--model", str(model_path), "--artifact-dir", str(artifacts_dir)],
        check=True,
    )

    assert (artifacts_dir / "model.joblib").exists()
    assert (artifacts_dir / "preprocessor.joblib").exists()

    # 4) run evaluate
    subprocess.run(
        [sys.executable, "-m", "risk_system.cli", "evaluate", "--base", str(base_path), "--policy", str(policy_path), "--artifact-dir", str(artifacts_dir)],
        check=True,
    )

    assert (artifacts_dir / "eval_metrics.json").exists()
