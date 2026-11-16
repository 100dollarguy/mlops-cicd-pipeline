import argparse
import sys

from mlflow.exceptions import MlflowException

from scripts.model_utils import (
    MODEL_NAME,
    get_mlflow_client,
    validate_model_version,
)

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"


def promote_model(
    version: str,
    approve: bool,
    expected_current: str,
) -> None:
    client = get_mlflow_client()

    staging = client.get_model_version_by_alias(
        MODEL_NAME, STAGING_ALIAS
    )

    if staging.version != version:
        raise ValueError(
            f"Version {version} is not the current staging version. "
            f"Current staging version: {staging.version}"
        )

    candidate, accuracy, f1_score = validate_model_version(
        client, version
    )

    try:
        previous = client.get_model_version_by_alias(
            MODEL_NAME, PRODUCTION_ALIAS
        )
        previous_version = previous.version

    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise

        previous_version = None

    if previous_version != expected_current:
        raise ValueError(
            f"Production currently points to {previous_version}, "
            f"not expected version {expected_current}. "
            "Refusing to overwrite production."
        )

    if previous_version == version:
        raise ValueError(
            f"Version {version} is already in production."
        )

    print(f"Candidate: {MODEL_NAME} version {version}")
    print(f"Status: {candidate.status}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(
        f"Previous production version: "
        f"{previous_version or 'None'}"
    )

    if not approve:
        print("\nDRY RUN: No production changes made.")
        print("Re-run with --approve to promote this version.")
        return

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=version,
    )

    print(
        f"\nProduction promotion successful: "
        f"{MODEL_NAME} version {version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a validated staging model to production."
    )

    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-current", required=True)

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve and execute production promotion.",
    )

    args = parser.parse_args()

    try:
        promote_model(
            args.version,
            args.approve,
            args.expected_current,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
