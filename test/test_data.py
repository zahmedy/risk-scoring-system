from risk_system.data import split_X_y, infer_schema
from risk_system.utils import load_config

import pytest
import pandas as pd

cfg = load_config()

def test_split_X_y_raises_when_target_missing():
    df = pd.DataFrame({
        "age": [20, 30],
        "income": [50000, 60000],
    })

    with pytest.raises(ValueError) as excinfo:
        split_X_y(df, target="class")

    assert "class" in str(excinfo.value)

def test_infer_schema_dtypes_are_strings():
    df = pd.DataFrame({
        "age": [20, 30],
        "income": [50000, 60000],
        "class": [0, 1],
    })

    schema = infer_schema(df, target="class")

    assert isinstance(schema, dict)
    assert "dtypes" in schema

    for dtype in schema["dtypes"].values():
        assert isinstance(dtype, str)

    
