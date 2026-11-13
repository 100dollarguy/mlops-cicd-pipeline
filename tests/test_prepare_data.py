import pandas as pd

from src.prepare_data import prepare_dataset


def test_prepare_dataset(tmp_path, monkeypatch):
    import src.prepare_data as prepare_module

    monkeypatch.setattr(
        prepare_module,
        "RAW_DATA_DIR",
        tmp_path / "raw",
    )

    prepare_dataset()

    train_path = tmp_path / "raw" / "train.csv"
    test_path = tmp_path / "raw" / "test.csv"

    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) == 455
    assert len(test_df) == 114
    assert "target" in train_df.columns
    assert "target" in test_df.columns
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0
