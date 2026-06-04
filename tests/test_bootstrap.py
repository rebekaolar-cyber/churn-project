"""Tests for the deploy-time artifact bootstrap."""

from __future__ import annotations

from churn import bootstrap, config


def test_ensure_artifacts_builds_everything_from_raw(tmp_paths):
    # Only the raw CSV exists in tmp_paths; nothing processed or trained yet.
    assert not config.X_TRAIN_FILE.exists()
    assert not config.MODEL_FILE.exists()

    bootstrap.ensure_artifacts()

    assert config.X_TRAIN_FILE.exists()
    assert config.MODEL_FILE.exists()
    assert config.FEATURES_FILE.exists()


def test_ensure_artifacts_is_idempotent(tmp_paths):
    bootstrap.ensure_artifacts()
    model_mtime = config.MODEL_FILE.stat().st_mtime

    # Second call should be a no-op (artifacts already present).
    bootstrap.ensure_artifacts()
    assert config.MODEL_FILE.stat().st_mtime == model_mtime
