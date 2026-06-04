"""Shared fixtures: synthetic Telco-style data and tmp-dir path redirection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small synthetic dataset mirroring the Telco churn schema.

    60 rows with both churn classes well represented so a stratified split and
    SMOTE (k_neighbors=5) both have enough minority samples to work.
    """
    rng = np.random.default_rng(0)
    n = 60
    genders = rng.choice(["Female", "Male"], n)
    yes_no = lambda: rng.choice(["No", "Yes"], n)  # noqa: E731
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    opt = lambda: rng.choice(["No", "Yes", "No internet service"], n)  # noqa: E731

    df = pd.DataFrame(
        {
            "customerID": [f"ID-{i:04d}" for i in range(n)],
            "gender": genders,
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": yes_no(),
            "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": opt(),
            "OnlineBackup": opt(),
            "DeviceProtection": opt(),
            "TechSupport": opt(),
            "StreamingTV": opt(),
            "StreamingMovies": opt(),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            # Mix of numeric strings and blanks to exercise coercion + imputation.
            "TotalCharges": [
                "" if i % 10 == 0 else f"{rng.uniform(20, 8000):.2f}" for i in range(n)
            ],
            # Balanced target: alternating Yes/No guarantees 30 of each.
            "Churn": ["Yes" if i % 2 == 0 else "No" for i in range(n)],
        }
    )
    return df


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch, raw_df):
    """Redirect all config paths into a temp dir and write the synthetic raw CSV.

    Returns the temp root so tests can assert on produced files.
    """
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    models_dir = tmp_path / "models"
    raw_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    raw_file = raw_dir / "telco_churn.csv"
    raw_df.to_csv(raw_file, index=False)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "RAW_DATA_FILE", raw_file)
    monkeypatch.setattr(config, "X_TRAIN_FILE", processed_dir / "X_train.csv")
    monkeypatch.setattr(config, "X_TEST_FILE", processed_dir / "X_test.csv")
    monkeypatch.setattr(config, "Y_TRAIN_FILE", processed_dir / "y_train.csv")
    monkeypatch.setattr(config, "Y_TEST_FILE", processed_dir / "y_test.csv")
    monkeypatch.setattr(config, "MODEL_FILE", models_dir / "rf_model.joblib")
    monkeypatch.setattr(config, "FEATURES_FILE", models_dir / "features.joblib")
    return tmp_path
