# Risk Scoring System
End-to-end ML risk assessment system for tabular data. Current focus: simple preprocessing, Random Forest baseline, and exploratory notebooks for model selection, cost tuning, and drift investigation.

## Project layout
```
risk-scoring-system/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ base.yaml              # dataset path/target/split settings
│  └─ model.yaml             # model type + hyperparameters
├─ data/
│  ├─ raw/                   # local only (gitignored)
│  └─ processed/             # cached cleaned splits (optional)
├─ artifacts/                # serialized models, schema, metrics
├─ notebooks/
│  ├─ 01_eda.ipynb           # exploratory data analysis
│  ├─ 02_baseline_lr.ipynb   # logistic regression baseline
│  ├─ 03_rf_comparison.ipynb # Random Forest comparison
│  ├─ 04_threshold_costs.ipynb # decision thresholds vs. costs
│  └─ 05_drift_checks.ipynb  # data/prediction drift checks
├─ src/risk_system/          # library code (data, preprocess, train, etc.)
├─ api/                      # FastAPI scaffold (predict endpoint WIP)
└─ test/                     # unit tests
```

## Prerequisites
- Python 3.9+
- Recommended: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies (core + dev):  
  `pip install pandas numpy scikit-learn pyyaml joblib fastapi uvicorn pytest`
  - Once dependency metadata is added to `pyproject.toml`, prefer `pip install -e ".[dev]"`.

## Configuration
- `configs/base.yaml`: dataset location, target column, positive label, split settings.
- `configs/model.yaml`: model type (`rf` for now), implementation (`scratch`), and RF hyperparameters.
- Update these before running notebooks or any training script so paths and targets match your data.

## Data expectations
- Place the training CSV at `data/raw/risk.csv` (configurable).  
  - Must include the target column defined in `configs/base.yaml` (`default` by default).
  - Numeric/categorical features are inferred automatically in `risk_system.preprocess`.
- Optional: write cleaned/train-test splits to `data/processed/` if you add that step to your workflow.

## Running the workflow
The library code is callable from notebooks or your own scripts:
- Data loading/splitting: `risk_system.data.load_csv`, `split_X_y`, `train_test_split_df`.
- Preprocessing: `risk_system.preprocess.make_preprocessor` (impute + encode + scale if needed).
- Model training: `risk_system.train.train(cfg_base, cfg_model)` (Random Forest scaffold; extend to persist artifacts/metrics).
- Monitoring utilities are stubbed in `risk_system.monitor` for future drift/prediction checks.

### Notebooks
- Start Jupyter: `python -m pip install notebook` (if needed) then `jupyter notebook notebooks/`.
- Notebook purposes:
  - `01_eda.ipynb`: understand feature distributions/quality.
  - `02_baseline_lr.ipynb`: quick linear baseline.
  - `03_rf_comparison.ipynb`: compare RF performance to baseline.
  - `04_threshold_costs.ipynb`: pick decision thresholds by cost/recall trade-offs.
  - `05_drift_checks.ipynb`: check data/prediction drift across time slices; add your data snapshots before running.

### Tests
- Run unit tests: `pytest test`.

## API (scaffold)
- `api/main.py` is a FastAPI placeholder for a `/predict` endpoint; wire it to the trained model/preprocessor artifacts under `artifacts/` when available.
- Typical run command after implementing `app`: `uvicorn api.main:app --reload`.

## Next steps / contributions
- Wire `risk_system.train` to persist artifacts and metrics to `artifacts/`.
- Implement evaluation/explainability in `evaluate.py` and `explain.py`.
- Finalize the FastAPI app, request/response schemas, and load artifacts for online scoring.
- Add dependency pins to `pyproject.toml` once the pipeline stabilizes.
