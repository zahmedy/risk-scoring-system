import joblib 
from risk_system.utils import load_config

_MODEL = None
_PREPROCESSOR = None
_CFG = None

def load_artifacts(prepro_path: str = "artifacts/preprocessor.joblib", 
                   model_path: str = "artifacts/model.joblib", 
                   cfg_path: str = "configs/base.yaml"):
    global _MODEL, _PREPROCESSOR, _CFG
    _MODEL = joblib.load(model_path)
    _PREPROCESSOR = joblib.load(prepro_path)
    _CFG = load_config(cfg_path)

    return _MODEL, _PREPROCESSOR, _CFG

def get_artifacts():
    if _MODEL is None or _PREPROCESSOR is None or _CFG is None:
        raise RuntimeError("Artifacts not loaded. Call load_artifacts() at startup.")
    return _MODEL, _PREPROCESSOR, _CFG