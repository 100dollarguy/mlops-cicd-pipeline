
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "target"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "breast-cancer-classification"
REGISTERED_MODEL_NAME = "BreastCancerClassifier"


def train_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    parameters = {
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "scaler": "StandardScaler",
        "test_size": 0.2,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="logistic-regression") as run:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
        )

        report_path = REPORT_DIR / "classification_report.json"

        with open(report_path, "w") as file:
            json.dump(report, file, indent=4)

        metrics_path = REPORT_DIR / "metrics.json"

        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        joblib.dump(model, MODEL_DIR / "model.joblib")

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="breast-cancer-model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        print("\nMLflow Run ID:", run.info.run_id)

        print("\nModel Evaluation")
        print("----------------")
        print(json.dumps(metrics, indent=4))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, predictions))

        print("\nModel registered:", REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    train_model()

