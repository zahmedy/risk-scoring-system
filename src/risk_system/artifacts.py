from risk_system.utils import load_config

import joblib 

def load_artifacts(prepro_path: str = "artifacts/preprocessor.joblib", 
                   model_path: str = "artifacts/model.joblib", 
                   cfg_path: str = "configs/base.yaml"
):
    MODEL = joblib.load(model_path)
    PREPROCESSOR = joblib.load(prepro_path)
    CFG = load_config(cfg_path)

    return MODEL, PREPROCESSOR, CFG