import yaml
from pathlib import Path

def load_yaml(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {path}")

    return config

def merge_dicts(base, override) -> dict:
    return base | override


if __name__ == "__main__":
    cfg = load_yaml("configs/base.yaml")
    print(cfg)