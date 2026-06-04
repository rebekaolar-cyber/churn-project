"""Preprocessing for the Telco customer-churn dataset.

The pipeline loads the raw CSV, cleans and encodes it, performs a stratified
train/test split, and persists the four resulting frames to ``data/processed``.
Each step is a small, independently testable function.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw churn CSV.

    Raises:
        FileNotFoundError: if the raw dataset is missing.
    """
    source = Path(config.RAW_DATA_FILE if path is None else path)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw data not found at {source}. Place telco_churn.csv under data/raw/."
        )
    df = pd.read_csv(source)
    logger.info("Loaded %d rows and %d columns", len(df), df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``TotalCharges`` to numeric, impute its NaNs with the median, and
    drop the customer-ID column. Returns a new frame; the input is not mutated.

    Raises:
        KeyError: if a required column is missing.
    """
    required = {config.NUMERIC_TEXT_COLUMN, config.ID_COLUMN, config.TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df[config.NUMERIC_TEXT_COLUMN] = pd.to_numeric(
        df[config.NUMERIC_TEXT_COLUMN], errors="coerce"
    )
    median = df[config.NUMERIC_TEXT_COLUMN].median()
    df[config.NUMERIC_TEXT_COLUMN] = df[config.NUMERIC_TEXT_COLUMN].fillna(median)
    logger.info("Imputed %s NaNs with median %.2f", config.NUMERIC_TEXT_COLUMN, median)

    return df.drop(columns=[config.ID_COLUMN])


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map the target to 0/1 and one-hot encode categoricals (drop_first=True).

    Returns:
        A ``(X, y)`` tuple of the encoded feature matrix and integer target.
    """
    y = df[config.TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    features = df.drop(columns=[config.TARGET_COLUMN])

    categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
    X = pd.get_dummies(
        features, columns=categorical_cols, drop_first=True, dtype=int
    )
    logger.info("Encoded %d categorical columns into %d features",
                len(categorical_cols), X.shape[1])
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split using the configured size and seed."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist the four processed frames to ``data/processed``."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(config.X_TRAIN_FILE, index=False)
    X_test.to_csv(config.X_TEST_FILE, index=False)
    y_train.to_csv(config.Y_TRAIN_FILE, index=False)
    y_test.to_csv(config.Y_TEST_FILE, index=False)
    logger.info("Saved processed datasets to %s", config.PROCESSED_DATA_DIR)


def preprocess_data(
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run the full preprocessing pipeline end to end.

    Args:
        save: when True (default), also write the splits to ``data/processed``.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    df = load_raw_data()
    df = clean_data(df)
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info(
        "Split into %d train / %d test samples (%d features)",
        len(X_train), len(X_test), X.shape[1],
    )
    if save:
        save_processed(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preprocess_data()


if __name__ == "__main__":
    main()
