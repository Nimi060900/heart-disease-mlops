import pytest

from src import data
from src.features import FEATURES, build_pipeline


@pytest.fixture(scope="session")
def raw_df():
    return data.load_raw(data.download())


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return data.clean(raw_df)


@pytest.fixture(scope="session")
def fitted_pipeline(clean_df):
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    return pipe.fit(clean_df[FEATURES], clean_df["target"])
