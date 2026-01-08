# Risk Scoring System (Production ML API)

End-to-end risk scoring system packaged as a production-ready ML inference API.
Includes schema-based validation, deterministic score mapping, decision thresholds, and AWS-ready container deployment.

## Highlights
- FastAPI inference service with `/health`, `/version`, and `/score`
- Startup-time artifact loading (model + preprocessor + schema)
- Schema-driven input validation with sanitized 400/500 error handling
- Probability → score (300–850) mapping + approve/review/decline decisioning
- Dockerized deployment (AWS App Runner compatible)
- Optional API key protection via `x-api-key` (enabled when `API_KEY` is set)

## Example Request
```json
{
  "request_id": "demo-001",
  "applicant": {
    "checking_status": "'<0'",
    "duration": 6,
    "credit_history": "'critical/other existing credit'",
    "purpose": "radio/tv",
    "credit_amount": 1169,
    "savings_status": "'no known savings'",
    "employment": "'>=7'",
    "installment_commitment": 4,
    "personal_status": "'male single'",
    "other_parties": "none",
    "residence_since": 4,
    "property_magnitude": "'real estate'",
    "age": 67,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 2,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes"
  }
}
```

## Example Response
```json
{
  "request_id": "demo-001",
  "score": 697,
  "decision": "review",
  "probability_default": 0.1253,
  "latency_ms": 6.07
}
```

## Architecture

Client
│
▼
FastAPI (/score)
│
├─ API key check (optional)
├─ Schema validation (required fields, types)
│
▼
Preprocessor (joblib)
│
▼
Model inference
│
▼
Probability of Default (PD)
│
├─ PD → credit-style score (300–850)
└─ Decision thresholds (approve / review / decline)
│
▼
JSON response (+ latency)

**Design goals**
- Deterministic inference (no side effects)
- Fail fast on invalid input
- Startup-time loading for low per-request latency
- Cloud-friendly (container + health checks)

## Repository Structure

risk-scoring-system/
- `artifacts/` — trained model, preprocessor, schema, metrics, drift baselines
- `configs/` — data/model/monitor/policy configs
- `src/risk_system/` — training, preprocessing, evaluation, service, CLI, API
- `notebooks/` — EDA, baselines, RF comparison, threshold tuning, drift checks
- `data/` — local raw/processed datasets (gitignored)
- `test/` — unit tests
- `Dockerfile` — container image for serving
- `pyproject.toml` — package metadata and dependencies

**Why this matters**
- Clear separation between **API**, **ML logic**, and **artifacts**
- Custom models are packaged and importable (pickle-safe)
- No training code in the inference path

## Model & Data

- Model: Random Forest classifier implemented from scratch
- Target: binary risk outcome (good / bad)
- Output: calibrated probability of default (PD)
- Decisioning:
  - PD ≤ approve threshold → approve
  - PD ≤ review threshold → review
  - else → decline

The API exposes both the raw probability and a credit-style score for downstream consumers.

## Run Locally (Docker)

```bash
docker build -t risk-scoring-api .
docker run -p 8080:8080 risk-scoring-api
```

---

## Deployment (AWS App Runner)

```md
## Deployment (AWS)

The service is containerized and deployed using **AWS App Runner**.

Deployment flow:
1. Build Docker image
2. Push image to Amazon ECR
3. Deploy App Runner service on port `8080`
4. Health checks via `/health`

The service supports environment-based configuration and optional API key protection via `x-api-key`.
```

## Reliability & Lessons Learned

- **Schema-driven validation** prevents malformed requests from reaching the model
- **Sanitized error handling** (400 vs 500) for safe client responses
- **Startup-time artifact loading** avoids per-request I/O overhead
- **API key authentication** enabled via environment variables
- Resolved real production issues:
  - scikit-learn pickle version mismatches
  - custom model import paths in containers
  - ARM vs x86_64 container architecture on AWS

