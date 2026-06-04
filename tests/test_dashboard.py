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
