"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import config, data_prep


def test_load_raw_data_success(tmp_paths):
    df = data_prep.load_raw_data()
    assert len(df) == 60
    assert config.ID_COLUMN in df.columns


def test_load_raw_data_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw data not found"):
        data_prep.load_raw_data(tmp_path / "nope.csv")


def test_clean_data_coerces_imputes_and_drops_id(raw_df):
    cleaned = data_prep.clean_data(raw_df)
    assert config.ID_COLUMN not in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned[config.NUMERIC_TEXT_COLUMN])
    assert cleaned[config.NUMERIC_TEXT_COLUMN].isna().sum() == 0


def test_clean_data_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    data_prep.clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_data_missing_column_raises(raw_df):
    broken = raw_df.drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(KeyError, match="missing required columns"):
        data_prep.clean_data(broken)


def test_encode_features_maps_target_and_one_hot(raw_df):
    cleaned = data_prep.clean_data(raw_df)
    X, y = data_prep.encode_features(cleaned)
    assert set(y.unique()) <= {0, 1}
    assert config.TARGET_COLUMN not in X.columns
    # All encoded columns must be numeric for the model.
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert X.shape[0] == len(raw_df)


def test_split_data_proportions_and_stratification(raw_df):
    cleaned = data_prep.clean_data(raw_df)
    X, y = data_prep.encode_features(cleaned)
    X_train, X_test, y_train, y_test = data_prep.split_data(X, y)
    assert len(X_test) == pytest.approx(len(X) * config.TEST_SIZE, abs=1)
    assert len(X_train) + len(X_test) == len(X)
    # Stratification keeps both classes in the test split.
    assert set(y_test.unique()) == {0, 1}


def test_save_processed_writes_four_files(tmp_paths, raw_df):
    cleaned = data_prep.clean_data(raw_df)
    X, y = data_prep.encode_features(cleaned)
    X_train, X_test, y_train, y_test = data_prep.split_data(X, y)
    data_prep.save_processed(X_train, X_test, y_train, y_test)
    for f in (
        config.X_TRAIN_FILE,
        config.X_TEST_FILE,
        config.Y_TRAIN_FILE,
        config.Y_TEST_FILE,
    ):
        assert f.exists()


def test_preprocess_data_end_to_end_saves(tmp_paths):
    X_train, X_test, y_train, y_test = data_prep.preprocess_data(save=True)
    assert len(X_train) > len(X_test)
    assert config.X_TRAIN_FILE.exists()
    assert isinstance(X_train, pd.DataFrame)


def test_preprocess_data_no_save(tmp_paths):
    data_prep.preprocess_data(save=False)
    assert not config.X_TRAIN_FILE.exists()


def test_main_runs(tmp_paths):
    data_prep.main()
    assert config.X_TRAIN_FILE.exists()
