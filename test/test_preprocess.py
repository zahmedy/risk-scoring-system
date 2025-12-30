import pandas as pd
from risk_system.preprocess import infer_feature_types, make_preprocessor

def test_preprocessor():
    df = pd.DataFrame({
        "Age": [20, 40, None],
        "income": [1.1, None, 1.2],
        "job": ["ML Engineer", "AI Engineer", None]
    })

    numerical_features, categorical_features = infer_feature_types(df)

    preprocessor = make_preprocessor(numeric_features=numerical_features, 
                                     categorical_features=categorical_features, 
                                     model_type="lr")
    
    new_df = preprocessor.fit_transform(df)

    assert hasattr(new_df, "shape")
    assert df.shape[0] == new_df.shape[0]