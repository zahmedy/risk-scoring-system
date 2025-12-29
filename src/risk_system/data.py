import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_csv(cfg) -> pd.DataFrame:
    try:
        df = pd.read_csv(cfg["dataset"]["path"])
    except:
        raise FileNotFoundError(f"File not found in path {cfg["dataset"]["path"]}")
    
    return df

def split_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target in df.columns.tolist():
        y = df[target]
        X = df.drop(columns=target)
    else:
        raise ValueError(f"Warning: Target is not in Dataframe, existing...")

    return X, y

def train_test_split_df(X, y, cfg):
    if cfg['split']['stratify']:
        if len(np.unique(y)) == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['split']['test_size'], random_state=cfg['split']['random_state'])
            raise Warning(f"Warning: stratify disabled only 1 class in y")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['split']['test_size'], random_state=cfg['split']['random_state'], stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['split']['test_size'], random_state=cfg['split']['random_state'])

    return X_train, X_test, y_train, y_test

def infer_schema(df: pd.DataFrame, target: str) -> dict:
     datatypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
     return {"columns": df.columns.tolist(),"dtypes": datatypes,"target": target}