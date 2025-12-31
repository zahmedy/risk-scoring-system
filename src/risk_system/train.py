from risk_system.preprocess import make_preprocessor, infer_feature_types
from risk_system.data import load_csv, split_X_y, train_test_split_df



def train(cfg_base: dict, cfg_model: dict) -> dict:
    df = load_csv(cfg_base)
    X, y = split_X_y(df, cfg_base["dataset"]["target"])
    X_train, X_test, y_train, y_test = train_test_split_df(X, y, cfg_base)
    num_features, cat_features = infer_feature_types(X_train)
    preprocessor = make_preprocessor(num_features, cat_features, model_type=cfg_model["type"])
    preprocessor.fit_transform(X_train, y_train)
    
