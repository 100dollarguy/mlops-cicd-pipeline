import json

from src.train import (
    load_data,
    evaluate_model,
    save_reports,
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_reports_are_generated():
    X_train, X_test, y_train, y_test = load_data()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    metrics, report, _ = evaluate_model(model, X_test, y_test)

    save_reports(metrics, report)

    from src.train import REPORT_DIR

    metrics_file = REPORT_DIR / "metrics.json"
    report_file = REPORT_DIR / "classification_report.json"

    assert metrics_file.exists()
    assert report_file.exists()

    with open(metrics_file) as file:
        saved_metrics = json.load(file)

    assert saved_metrics["accuracy"] == metrics["accuracy"]
