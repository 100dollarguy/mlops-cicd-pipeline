import math
import os

from mlflow import MlflowClient


MODEL_NAME = "BreastCancerClassifier"

MIN_ACCURACY = 0.95
MIN_F1_SCORE = 0.95


def get_mlflow_client() -> MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI must be explicitly configured."
        )

    return MlflowClient(tracking_uri=tracking_uri)


def validate_model_version(client, version: str):
    model = client.get_model_version(MODEL_NAME, version)

    if model.status != "READY":
        raise ValueError(
            f"Model version {version} is not READY: {model.status}"
        )

    tags = model.tags

    if tags.get("validation_status") != "passed":
        raise ValueError(
            f"Model version {version} has not passed validation."
        )

    try:
        accuracy = float(tags["validation_accuracy"])
        f1_score = float(tags["validation_f1_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Validation metrics are missing or invalid."
        ) from exc

    if not math.isfinite(accuracy) or not math.isfinite(f1_score):
        raise ValueError(
            "Validation metrics must be finite numbers."
        )

    if not 0 <= accuracy <= 1 or not 0 <= f1_score <= 1:
        raise ValueError(
            "Validation metrics must be between 0 and 1."
        )

    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Accuracy {accuracy} is below {MIN_ACCURACY}."
        )

    if f1_score < MIN_F1_SCORE:
        raise ValueError(
            f"F1-score {f1_score} is below {MIN_F1_SCORE}."
        )

    return model, accuracy, f1_score
