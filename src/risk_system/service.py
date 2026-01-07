import numpy as np
import pandas as pd

from risk_system.evaluate import predict_scores
from risk_system.artifacts import get_artifacts

def score_one(applicant: dict) -> dict:
    model, preprocessor, cfg_base = get_artifacts()

    df = pd.DataFrame([applicant])
    X = preprocessor.transform(df)
    
    prob_d = predict_scores(model, X)
    prob_d = float(prob_d[0])

    eps = cfg_base["score_mapping"]["eps"]
    offset = cfg_base["score_mapping"]["offset"]
    factor = cfg_base["score_mapping"]["factor"]

    pd_clamped = min(max(prob_d, eps), 1-eps)
    odds = (1 - pd_clamped) / pd_clamped

    min_score = cfg_base["score_mapping"]["min_score"]
    max_score = cfg_base["score_mapping"]["max_score"]

    score = offset + factor * np.log(odds)
    score = max(min_score, min(score, max_score))
    score = int(round(score))

    approve_max_pd = cfg_base["decision_thresholds"]["approve_max_pd"]
    review_max_pd = cfg_base["decision_thresholds"]["review_max_pd"]

    if prob_d <= approve_max_pd:
        decision = "approve"
    elif prob_d <= review_max_pd:
        decision = "review"
    else:
        decision = "decline"

    return  {
        "score": score,                    # e.g., 300-850
        "decision": decision,                 # "approve" | "review" | "decline"
        "probability_default": prob_d,    # 0..1
    }