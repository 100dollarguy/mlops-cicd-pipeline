from pathlib import Path
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

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
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "target"
EXPERIMENT_NAME = "breast-cancer-classification"


def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or testing dataset is missing."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    validate_data(train_df, test_df)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def validate_data(train_df, test_df):
    for name, df in [
        ("training", train_df),
        ("testing", test_df),
    ]:
        if TARGET not in df.columns:
            raise ValueError(
                f"{name} dataset is missing the target column."
            )

        if df.empty:
            raise ValueError(
                f"{name} dataset is empty."
            )

        if df.isnull().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

        if set(df[TARGET].unique()) != {0, 1}:
            raise ValueError(
                f"{name} dataset must contain target labels 0 and 1."
            )

    train_features = set(train_df.columns) - {TARGET}
    test_features = set(test_df.columns) - {TARGET}

    if train_features != test_features:
        raise ValueError(
            "Training and testing feature columns do not match."
        )


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return metrics, report, confusion_matrix(
        y_test,
        predictions,
    )


def save_reports(metrics, report):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "metrics.json"
    report_path = REPORT_DIR / "classification_report.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)


def train_model():
    X_train, X_test, y_train, y_test = load_data()

    parameters = {
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "scaler": "StandardScaler",
        "test_size": 0.2,
        "random_state": 42,
    }

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    metrics, report, matrix = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_reports(metrics, report)

    # MLflow experiment tracking
    mlflow_enabled = (
        os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
    )

    if mlflow_enabled:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(
            run_name="logistic-regression"
        ) as run:

            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)

            mlflow.log_artifact(
                str(REPORT_DIR / "metrics.json")
            )

            mlflow.log_artifact(
                str(REPORT_DIR / "classification_report.json")
            )

            # Explicit model input and output signature
            input_example = X_train.head(5).astype("float64")

            signature = infer_signature(
                input_example,
                model.predict(input_example),
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="breast-cancer-model",
                input_example=input_example,
                signature=signature,
            )

            print("\nMLflow Run ID:", run.info.run_id)

    print("\nModel Evaluation")
    print("----------------")
    print(json.dumps(metrics, indent=4))

    print("\nConfusion Matrix")
    print(matrix)

    return metrics


if __name__ == "__main__":
    train_model()
