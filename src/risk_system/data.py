import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

def load_csv(cfg) -> pd.DataFrame:
    path = cfg['dataset']['path']
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found in path {path}") from e
    
    return df

def split_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
            f"Available columns: {list(df.columns)}"
        )

    y = df[target]
    X = df.drop(columns=target)

    return (X, y)

def train_test_split_df(X, y, cfg):
    stratify = y if cfg["split"]["stratify"] and len(np.unique(y)) > 1 else None
    if cfg["split"]["stratify"] and stratify is None:
        warnings.warn("Stratify requested but y has only 1 class; proceeding without stratify.", RuntimeWarning)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=stratify
)

    return X_train, X_test, y_train, y_test

def infer_schema(df: pd.DataFrame, target: str) -> dict:
     datatypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
     return {"columns": df.columns.tolist(),"dtypes": datatypes, "target": target}