from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import load_data, evaluate_model


def test_model_training():
    X_train, X_test, y_train, y_test = load_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_model_evaluation():
    X_train, X_test, y_train, y_test = load_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    metrics, report, matrix = evaluate_model(model, X_test, y_test)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

    assert "accuracy" in report
    assert matrix.shape == (2, 2)


def test_train_model_without_mlflow(monkeypatch, tmp_path):
    import src.train as train_module

    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setattr(train_module, "REPORT_DIR", tmp_path)

    metrics = train_module.train_model()

    assert metrics["accuracy"] > 0.90
    assert metrics["f1_score"] > 0.90
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_report.json").exists()
