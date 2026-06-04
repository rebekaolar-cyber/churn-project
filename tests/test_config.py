"""Tests for path/constants configuration."""

from __future__ import annotations

import importlib
from pathlib import Path

from churn import config


def test_paths_are_paths():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.RAW_DATA_FILE, Path)
    assert config.MODEL_FILE.name == "rf_model.joblib"
    assert config.FEATURES_FILE.name == "features.joblib"


def test_derived_paths_nest_under_their_parents():
    assert config.RAW_DATA_FILE.parent == config.RAW_DATA_DIR
    assert config.X_TRAIN_FILE.parent == config.PROCESSED_DATA_DIR
    assert config.RAW_DATA_DIR.parent == config.DATA_DIR


def test_env_overrides_data_and_models_dirs(monkeypatch):
    monkeypatch.setenv("CHURN_DATA_DIR", "/tmp/churn-data")
    monkeypatch.setenv("CHURN_MODELS_DIR", "/tmp/churn-models")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATA_DIR == Path("/tmp/churn-data")
        assert reloaded.MODELS_DIR == Path("/tmp/churn-models")
        assert reloaded.RAW_DATA_FILE == Path("/tmp/churn-data/raw/telco_churn.csv")
    finally:
        # Restore module-level defaults for other tests.
        monkeypatch.delenv("CHURN_DATA_DIR")
        monkeypatch.delenv("CHURN_MODELS_DIR")
        importlib.reload(config)


def test_constants():
    assert config.TARGET_COLUMN == "Churn"
    assert 0 < config.TEST_SIZE < 1
    assert config.RANDOM_STATE == 42
