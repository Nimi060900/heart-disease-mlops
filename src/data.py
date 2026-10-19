"""Download, clean and save the UCI Heart Disease (Cleveland) dataset."""
import argparse
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"


def download(dest: Path = RAW_PATH, url: str = URL) -> Path:
    """Fetch the raw file from UCI unless it is already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target and impute missing values.

    Missing values only occur in `ca` and `thal` (categorical), so they are
    filled with the mode. Imputation is repeated inside the sklearn pipeline
    at training time, so this step only keeps the saved CSV complete.
    """
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    for col in df.columns.drop("target"):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in CATEGORICAL:
        df[col] = df[col].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true",
                        help="use the existing raw file only")
    args = parser.parse_args()
    raw = RAW_PATH if args.no_download else download()
    out = save_clean(clean(load_raw(raw)))
    print(f"Saved cleaned data to {out}")
