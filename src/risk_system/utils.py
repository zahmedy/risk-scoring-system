import yaml
import numpy as np

def to_dense(Xt) -> np.ndarray:
    if hasattr(Xt, "toarray"):
        return Xt.toarray()
    return Xt

def load_config(config_path="configs/base.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Config loaded: {config}")

    return config

    