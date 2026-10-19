"""FastAPI service exposing the heart-disease model."""
import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.predict import load_model, predict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[1])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=40, le=260, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=4, examples=[0])
    thal: int = Field(..., ge=3, le=7, examples=[6])


class Prediction(BaseModel):
    prediction: int
    confidence: float
    probability_disease: float


@app.middleware("http")
async def track(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    log.info("%s %s -> %s in %.1f ms", request.method, path,
             response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    try:
        load_model()
    except Exception as exc:  # model file missing or unreadable
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}")
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    result = predict([patient.model_dump()])[0]
    PREDICTIONS.labels(str(result["prediction"])).inc()
    log.info("prediction input=%s output=%s", patient.model_dump(), result)
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
