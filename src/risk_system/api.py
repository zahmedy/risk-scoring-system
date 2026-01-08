from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional 
import time
import logging
from requests import Request

from risk_system.artifacts import load_artifacts, get_artifacts
from risk_system.service import score_one
from risk_system.exceptions import BadRequestError

logger = logging.getLogger("risk_api")


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

@app.exception_handler(BadRequestError)
async def bad_request_handler(request: Request, exc: BadRequestError):
    logger.info("Bad request: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=400,
        content={
            "error": "bad_request",
            "message": "Invalid applicant payload.",
            "missing_fields": exc.missing_features,
        }
    )

@app.exception_handler((KeyError, ValueError, TypeError))
async def bad_input_handler(request: Request, exc: Exception):
    logger.info("Bad input: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=400,
        content={
            "error": "bad_request",
            "message": "Invalid applicant payload.",
        }
    )

@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Internal server error.",
        }
    )


@app.get("/version")
def get_version():
    _, _, CFG, _ = get_artifacts()
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
    
    result = score_one(req.applicant)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "request_id": req.request_id,
        **result,
        "latency_ms": latency_ms
    }
    
