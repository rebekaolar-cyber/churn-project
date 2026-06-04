"""Tests for the dashboard's pure helper functions (no Streamlit runtime)."""

from __future__ import annotations

import numpy as np
import pytest

from churn import dashboard


def test_build_input_row_orders_and_fills():
    features = ["a", "b", "c"]
    row = dashboard.build_input_row(features, {"a": 1, "c": 3})
    assert list(row.columns) == features
    assert row.iloc[0].tolist() == [1, 0, 3]  # missing "b" defaults to 0


@pytest.mark.parametrize(
    "proba,expected",
    [(0.0, "Low"), (0.32, "Low"), (0.5, "Medium"), (0.65, "Medium"), (0.9, "High")],
)
def test_risk_level(proba, expected):
    label, color = dashboard.risk_level(proba)
    assert label == expected
    assert color in {"green", "orange", "red"}


def test_top_factors_returns_sorted_topn():
    importances = np.array([0.1, 0.5, 0.3, 0.05])
    names = ["w", "x", "y", "z"]
    result = dashboard.top_factors(importances, names, n=2)
    assert [name for name, _ in result] == ["x", "y"]
    assert result[0][1] == pytest.approx(0.5)


# Full set of one-hot feature names produced by the training pipeline.
FEATURE_NAMES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def test_encode_inputs_high_risk_profile():
    sel = dashboard.PRESETS["⚠️ High-risk customer"]
    enc = dashboard.encode_inputs(sel, FEATURE_NAMES)

    # Every feature is present and the vector is fully numeric.
    assert set(enc) == set(FEATURE_NAMES)
    # Numeric passthroughs.
    assert enc["tenure"] == 2
    assert enc["MonthlyCharges"] == pytest.approx(95.0)
    assert enc["SeniorCitizen"] == 1
    # One-hot flags.
    assert enc["InternetService_Fiber optic"] == 1
    assert enc["InternetService_No"] == 0
    assert enc["PaymentMethod_Electronic check"] == 1
    assert enc["PaymentMethod_Mailed check"] == 0
    assert enc["Contract_One year"] == 0
    assert enc["Contract_Two year"] == 0  # Month-to-month is the dropped baseline


def test_encode_inputs_no_phone_and_no_internet():
    sel = dict(dashboard.DEFAULTS)
    sel["PhoneService"] = "No"
    sel["MultipleLines"] = "No phone service"
    sel["InternetService"] = "No"
    for field in dashboard.ADDON_FIELDS:
        sel[field] = "No internet service"

    enc = dashboard.encode_inputs(sel, FEATURE_NAMES)

    assert enc["PhoneService_Yes"] == 0
    assert enc["MultipleLines_No phone service"] == 1
    assert enc["MultipleLines_Yes"] == 0
    assert enc["InternetService_No"] == 1
    for field in dashboard.ADDON_FIELDS:
        assert enc[f"{field}_No internet service"] == 1
        assert enc[f"{field}_Yes"] == 0
