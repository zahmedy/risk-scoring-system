from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional 

from risk_system.artifacts import load_artifacts
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

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/score")
def score(req: ScoreRequest):
    try:
        result = score_one(req.applicant)
        return {
            "request_id": req.request_id,
            **result,
        }
    except Exception as e:
        # later we’ll narrow this into 400 vs 500 errors
        raise HTTPException(status_code=500, detail=str(e))