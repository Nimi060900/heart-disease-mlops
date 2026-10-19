"""Inference helper: load the saved pipeline and score patient records."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(records: list, model=None) -> list:
    """Return prediction (0/1) and confidence for each record (list of dicts)."""
    model = model or load_model()
    X = pd.DataFrame(records)[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return [
        {
            "prediction": int(p >= 0.5),
            "probability_disease": round(float(p), 4),
            "confidence": round(float(max(p, 1 - p)), 4),
        }
        for p in proba
    ]
