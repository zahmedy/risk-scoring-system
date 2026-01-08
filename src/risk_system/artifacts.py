import joblib 
import json
from risk_system.utils import load_config

_MODEL = None
_PREPROCESSOR = None
_CFG = None
_SCHEMA = None

def load_artifacts(prepro_path: str = "artifacts/preprocessor.joblib", 
                   model_path: str = "artifacts/model.joblib", 
                   cfg_path: str = "configs/base.yaml",
                   schema_path: str = "artifacts/schema.json"):
    global _MODEL, _PREPROCESSOR, _CFG, _SCHEMA
    _MODEL = joblib.load(model_path)
    _PREPROCESSOR = joblib.load(prepro_path)
    _CFG = load_config(cfg_path)
    with open(schema_path) as f:
        _SCHEMA = json.load(f)

    return _MODEL, _PREPROCESSOR, _CFG, _SCHEMA

def get_artifacts():
    if _MODEL is None or _PREPROCESSOR is None or _CFG is None or _SCHEMA is None:
        raise RuntimeError("Artifacts not loaded. Call load_artifacts() at startup.")
    return _MODEL, _PREPROCESSOR, _CFG, _SCHEMA