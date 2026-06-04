"""Central configuration: filesystem paths and pipeline constants.

Paths can be overridden with the ``CHURN_DATA_DIR`` and ``CHURN_MODELS_DIR``
environment variables so the package runs identically from a notebook, a CI
runner, or a deployed container regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: src/churn/config.py -> churn -> src -> repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = Path(os.environ.get("CHURN_MODELS_DIR", PROJECT_ROOT / "models"))

RAW_DATA_FILE: Path = RAW_DATA_DIR / "telco_churn.csv"

X_TRAIN_FILE: Path = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE: Path = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE: Path = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE: Path = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_FILE: Path = MODELS_DIR / "rf_model.joblib"
FEATURES_FILE: Path = MODELS_DIR / "features.joblib"

# Column semantics for the Telco churn dataset.
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"
# Stored as text in the raw CSV (blank for never-billed customers); coerced to numeric.
NUMERIC_TEXT_COLUMN: str = "TotalCharges"

# Reproducibility / split parameters.
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
