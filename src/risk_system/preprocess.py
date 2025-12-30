import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    numeric if dtype is number (int, float)

    categorical otherwise (object, category, bool)
    """
    if X.empty:
        raise ValueError("Cannot infer feature types from empty DataFrame")

    numerical = []
    categorical = []

    for col in X.columns:
        if is_numeric_dtype(X[col]):
            numerical.append(col)
        else:
            categorical.append(col)

    return numerical, categorical

def make_preprocessor(numeric_features, categorical_features, model_type: str):
    if model_type not in {"lr", "rf"}:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if model_type == "lr":
        num_pip = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        num_pip = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pip, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,   
    )

    return preprocessor