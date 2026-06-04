"""Tests for the training pipeline."""

from __future__ import annotations

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from churn import config, data_prep, train


@pytest.fixture
def processed(tmp_paths):
    """Produce and persist processed splits, then return them."""
    return data_prep.preprocess_data(save=True)


def test_load_processed_missing_raises(tmp_paths):
    with pytest.raises(FileNotFoundError, match="Processed data not found"):
        train.load_processed()


def test_load_processed_success(processed):
    X_train, X_test, y_train, y_test = train.load_processed()
    assert len(X_train) > 0
    assert set(y_train.unique()) <= {0, 1}


def test_balance_classes_equalizes(processed):
    X_train, _, y_train, _ = train.load_processed()
    X_res, y_res = train.balance_classes(X_train, y_train)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)


def test_build_model_returns_fitted_rf(processed):
    X_train, _, y_train, _ = train.load_processed()
    X_res, y_res = train.balance_classes(X_train, y_train)
    model = train.build_model(X_res, y_res)
    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X_res.head())
    assert len(preds) == 5


def test_build_model_with_tuning(processed):
    X_train, _, y_train, _ = train.load_processed()
    X_res, y_res = train.balance_classes(X_train, y_train)
    # Shrink the grid so tuning stays fast in CI.
    train.PARAM_GRID = {"n_estimators": [10], "max_depth": [3]}
    model = train.build_model(X_res, y_res, tune=True)
    assert isinstance(model, RandomForestClassifier)


def test_save_model_writes_artifacts(processed):
    X_train, _, y_train, _ = train.load_processed()
    model = train.build_model(X_train, y_train)
    train.save_model(model, X_train.columns.tolist())
    assert config.MODEL_FILE.exists()
    assert config.FEATURES_FILE.exists()
    assert joblib.load(config.FEATURES_FILE) == X_train.columns.tolist()


def test_train_model_end_to_end(processed):
    model, features = train.train_model()
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(features, list) and len(features) > 0
    assert config.MODEL_FILE.exists()


def test_main_runs(processed):
    train.main()
    assert config.MODEL_FILE.exists()
