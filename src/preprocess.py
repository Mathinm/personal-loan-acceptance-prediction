"""Data loading, cleaning, splitting and scaling for the Personal Loan dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Bank_Personal_Loan_Modelling.csv"

TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]          # identifier + high-cardinality noise
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle "
            "(teertha/personal-loan-modeling) and place it in data/."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID/ZIP and fix the 52 negative 'Experience' values (data-entry errors)."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["Experience"] = df["Experience"].clip(lower=0)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_test(X, y):
    """Stratified 80/20 split so both sets keep the ~9.6% positive rate."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def scale(X_train, X_test):
    """Fit StandardScaler on train only (no test leakage)."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_s, X_test_s, scaler


def get_data():
    """Convenience: load -> clean -> split. Returns raw (unscaled) splits."""
    df = clean_data(load_data())
    X, y = split_xy(df)
    return train_test(X, y)
