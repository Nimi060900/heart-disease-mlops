"""Preprocessing pipeline shared by training and inference."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC

FEATURES = NUMERIC + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numeric columns; mode-impute + one-hot categorical ones."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def build_pipeline(model) -> Pipeline:
    """Full pipeline: preprocessing followed by the estimator."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
