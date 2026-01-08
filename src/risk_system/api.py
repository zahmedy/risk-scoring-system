from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional 
import time

from risk_system.artifacts import load_artifacts, get_artifacts
from risk_system.service import score_one



class ScoreRequest(BaseModel):
    request_id: Optional[str] = None
    applicant: Dict[str, Any]



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
    except Exception as e:
        # Fail fast: if artifacts can't load, the service shouldn't run
        raise RuntimeError(f"Failed to load artifacts: {e}") from e
    yield

app = FastAPI(title="Risk Scoring API", version="0.1.0", lifespan=lifespan)

@app.get("/version")
def get_version():
    _, _, CFG = get_artifacts()
    meta = CFG.get("metadata", {})
    return {
        "api_version": app.version,
        "model_version": meta.get("model_version", "unknown"),
        "config_version": meta.get("config_version", "unknown"),
    }

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/score")
def score(req: ScoreRequest):
    t0 = time.perf_counter()
    try:
        result = score_one(req.applicant)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_id": req.request_id,
            **result,
            "latency_ms": latency_ms
        }
    except Exception as e:
        # later we’ll narrow this into 400 vs 500 errors
        raise HTTPException(status_code=500, detail=str(e))