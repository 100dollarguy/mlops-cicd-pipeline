import mlflow
from pathlib import Path

from src.train import train_model


def test_training_logs_to_mlflow(tmp_path, monkeypatch):
    database_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path}"
    report_dir = tmp_path / "reports"

    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    import src.train as train_module
    monkeypatch.setattr(train_module, "REPORT_DIR", report_dir)

    train_model()

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(
        "breast-cancer-classification"
    )

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) == 1
    assert runs[0].data.metrics["accuracy"] > 0.90
    assert runs[0].data.metrics["f1_score"] > 0.90
