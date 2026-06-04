"""Ensure model artifacts exist before the dashboard loads them.

On a fresh deploy (e.g. Streamlit Community Cloud) the repo ships the raw data
but not the trained model. This builds any missing artifacts from the raw CSV
so the app is self-sufficient without committing a large binary model.
"""

from __future__ import annotations

import logging

from churn import config, data_prep, train

logger = logging.getLogger(__name__)


def ensure_artifacts() -> None:
    """Generate processed splits and the trained model if they are missing."""
    processed_exists = all(
        f.exists()
        for f in (
            config.X_TRAIN_FILE,
            config.X_TEST_FILE,
            config.Y_TRAIN_FILE,
            config.Y_TEST_FILE,
        )
    )
    if not processed_exists:
        logger.info("Processed data missing; running preprocessing.")
        data_prep.preprocess_data(save=True)

    model_exists = config.MODEL_FILE.exists() and config.FEATURES_FILE.exists()
    if not model_exists:
        logger.info("Model missing; training from processed data.")
        train.train_model()
