import pandas as pd

from src import data


def test_raw_has_expected_columns(raw_df):
    assert list(raw_df.columns) == data.COLUMNS
    assert len(raw_df) == 999


def test_raw_missing_markers_parsed_as_nan(raw_df):
    assert raw_df[["ca", "thal"]].isna().sum().sum() == 6


def test_clean_has_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_target_is_binary(clean_df):
    assert set(clean_df["target"].unique()) == {0, 1}


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    data.clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_drops_exact_duplicates():
    row = {c: 1.0 for c in data.COLUMNS}
    df = pd.DataFrame([row, row])
    assert len(data.clean(df)) == 1


def test_save_and_load_roundtrip(clean_df, tmp_path):
    path = data.save_clean(clean_df, tmp_path / "clean.csv")
    pd.testing.assert_frame_equal(data.load_clean(path), clean_df, check_dtype=False)
