"""Train a Random Forest churn classifier on the processed data.

SMOTE balances the training set before fitting; the model and its feature
order are persisted to ``models/`` for the dashboard to consume.
"""

from __future__ import annotations

import logging
from typing import cast

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from churn import config

logger = logging.getLogger(__name__)

# Searched only when tune=True; kept small so a CV sweep stays affordable.
PARAM_GRID: dict[str, list[int | None]] = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2],
}


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the processed train/test splits.

    Raises:
        FileNotFoundError: if the processed files are missing (run preprocessing first).
    """
    required = [
        config.X_TRAIN_FILE,
        config.X_TEST_FILE,
        config.Y_TRAIN_FILE,
        config.Y_TEST_FILE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data not found: "
            + ", ".join(missing)
            + ". Run `python -m churn.data_prep` first."
        )
    X_train = pd.read_csv(config.X_TRAIN_FILE)
    X_test = pd.read_csv(config.X_TEST_FILE)
    y_train = cast(pd.Series, pd.read_csv(config.Y_TRAIN_FILE).squeeze("columns"))
    y_test = cast(pd.Series, pd.read_csv(config.Y_TEST_FILE).squeeze("columns"))
    return X_train, X_test, y_train, y_test


def balance_classes(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class with SMOTE."""
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info("SMOTE balanced training set to %d samples", len(X_res))
    return X_res, y_res


def build_model(
    X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False
) -> RandomForestClassifier:
    """Fit a Random Forest, optionally tuning hyperparameters via grid search."""
    if tune:
        search = GridSearchCV(
            RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            PARAM_GRID,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        logger.info("Best params: %s", search.best_params_)
        return search.best_estimator_

    model = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestClassifier, feature_names: list[str]) -> None:
    """Persist the model and feature order to ``models/``."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    joblib.dump(feature_names, config.FEATURES_FILE)
    logger.info("Saved model to %s", config.MODEL_FILE)


def train_model(
    tune: bool = False,
) -> tuple[RandomForestClassifier, list[str]]:
    """Run the full training pipeline and persist the artifacts.

    Returns:
        ``(model, feature_names)``.
    """
    X_train, X_test, y_train, y_test = load_processed()
    X_res, y_res = balance_classes(X_train, y_train)
    model = build_model(X_res, y_res, tune=tune)

    y_pred = model.predict(X_test)
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
    )

    feature_names = X_train.columns.tolist()
    save_model(model, feature_names)
    return model, feature_names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_model()


if __name__ == "__main__":
    main()
