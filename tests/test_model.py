import numpy as np
import pandas as pd

from src.features import FEATURES, build_preprocessor
from src.predict import predict


def test_preprocessor_output_is_finite_matrix(clean_df):
    out = build_preprocessor().fit_transform(clean_df[FEATURES])
    assert out.shape[0] == len(clean_df)
    assert np.isfinite(out.toarray() if hasattr(out, "toarray") else out).all()


def test_preprocessor_handles_missing_values(clean_df):
    X = clean_df[FEATURES].astype(float).copy()
    X.loc[0, ["chol", "ca"]] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out.toarray() if hasattr(out, "toarray") else out).any()


def test_pipeline_probabilities_valid(fitted_pipeline, clean_df):
    proba = fitted_pipeline.predict_proba(clean_df[FEATURES])
    assert proba.shape == (len(clean_df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_pipeline_beats_chance(fitted_pipeline, clean_df):
    acc = (fitted_pipeline.predict(clean_df[FEATURES]) == clean_df["target"]).mean()
    assert acc > 0.75


def test_predict_helper_output_format(fitted_pipeline, clean_df):
    records = clean_df[FEATURES].head(3).to_dict("records")
    out = predict(records, model=fitted_pipeline)
    assert len(out) == 3
    for r in out:
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0


def test_predict_is_deterministic(fitted_pipeline, clean_df):
    records = clean_df[FEATURES].head(5).to_dict("records")
    assert predict(records, model=fitted_pipeline) == predict(records, model=fitted_pipeline)


def test_unseen_category_does_not_crash(fitted_pipeline, clean_df):
    rec = clean_df[FEATURES].head(1).to_dict("records")[0]
    rec["thal"] = 99
    assert len(predict([rec], model=fitted_pipeline)) == 1
    assert isinstance(pd.DataFrame([rec]), pd.DataFrame)
