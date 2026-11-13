import pandas as pd
import pytest

from src.train import load_data, validate_data


def test_load_data():
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_validation_rejects_missing_target():
    train_df = pd.DataFrame({"feature": [1, 2]})
    test_df = pd.DataFrame({"feature": [3, 4]})

    with pytest.raises(ValueError, match="target column"):
        validate_data(train_df, test_df)


def test_validation_rejects_missing_values():
    train_df = pd.DataFrame({
        "feature": [1.0, None],
        "target": [0, 1],
    })
    test_df = pd.DataFrame({
        "feature": [2.0, 3.0],
        "target": [0, 1],
    })

    with pytest.raises(ValueError, match="missing values"):
        validate_data(train_df, test_df)


def test_validation_rejects_mismatched_features():
    train_df = pd.DataFrame({
        "feature_a": [1, 2],
        "target": [0, 1],
    })
    test_df = pd.DataFrame({
        "feature_b": [3, 4],
        "target": [0, 1],
    })

    with pytest.raises(ValueError, match="feature columns"):
        validate_data(train_df, test_df)
