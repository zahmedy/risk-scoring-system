# risk-scoring-system
End-to-end ML risk assessment system 


```risk-scoring-system/
├─ README.md
├─ pyproject.toml            
├─ .gitignore
├─ configs/
│  ├─ base.yaml              # dataset path, target name, features
│  └─ model.yaml             # RF params, threshold, etc.
├─ data/
│  ├─ raw/                   # local only (gitignored)
│  └─ processed/             # cached cleaned splits (optional)
├─ artifacts/
│  ├─ model.joblib
│  ├─ preprocessor.joblib
│  ├─ metrics.json
│  └─ schema.json
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_baseline_lr.ipynb
│  ├─ 03_rf_comparison.ipynb
│  ├─ 04_threshold_costs.ipynb
│  └─ 05_drift_checks.ipynb
├─ src/
│  └─ risk_system/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py             # load CSV, split, basic validation
│     ├─ preprocess.py       # impute, encode, scale (for LR)
│     ├─ train.py            # train + save artifacts
│     ├─ evaluate.py         # metrics + plots + metrics.json
│     ├─ explain.py          # feature importance, per-row explanation
│     ├─ monitor.py          # drift + prediction distribution checks
│     └─ utils.py
├─ api/
│  ├─ main.py                # FastAPI predict endpoint
│  └─ schema.py              # pydantic request/response
└─ tests/
   ├─ test_data.py
   ├─ test_preprocess.py
   └─ test_model_smoke.py
```