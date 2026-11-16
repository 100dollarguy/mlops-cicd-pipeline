import argparse
import sys
from datetime import datetime, timezone

from scripts.model_utils import (
    MODEL_NAME,
    get_mlflow_client,
    validate_model_version,
)

PRODUCTION_ALIAS = "production"


def rollback_model(
    target_version: str,
    expected_current: str,
    reason: str,
    approve: bool,
) -> None:
    if not reason.strip():
        raise ValueError("Rollback reason cannot be empty.")

    client = get_mlflow_client()

    current = client.get_model_version_by_alias(
        MODEL_NAME, PRODUCTION_ALIAS
    )

    if current.version != expected_current:
        raise ValueError(
            f"Production currently points to version {current.version}, "
            f"not expected version {expected_current}. "
            "Refusing to overwrite production."
        )

    if target_version == expected_current:
        raise ValueError(
            "Rollback target is already the production version."
        )

    target, accuracy, f1_score = validate_model_version(
        client, target_version
    )

    print(f"Current production: version {current.version}")
    print(f"Rollback target: version {target_version}")
    print(f"Target status: {target.status}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"Reason: {reason}")

    if not approve:
        print("\nDRY RUN: No production changes made.")
        print("Re-run with --approve to execute the rollback.")
        return

    rollback_time = datetime.now(timezone.utc).isoformat()

    # Record rollback intent before changing production.
    audit_tags = {
        "rollback_reason": reason,
        "rollback_timestamp": rollback_time,
        "rollback_from_version": expected_current,
        "rollback_status": "pending",
    }

    try:
        for key, value in audit_tags.items():
            client.set_model_version_tag(
                MODEL_NAME,
                target_version,
                key,
                value,
            )
    except Exception as exc:
        raise RuntimeError(
            "Rollback aborted: unable to record rollback intent. "
            "Production alias was not changed."
        ) from exc

    # Change production only after the intent is recorded.
    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=PRODUCTION_ALIAS,
            version=target_version,
        )
    except Exception as exc:
        try:
            client.set_model_version_tag(
                MODEL_NAME,
                target_version,
                "rollback_status",
                "failed",
            )
            client.set_model_version_tag(
                MODEL_NAME,
                target_version,
                "rollback_error",
                str(exc)[:1000],
            )
        except Exception as audit_exc:
            print(
                "WARNING: Alias update failed and failure audit "
                f"could not be completed: {audit_exc}",
                file=sys.stderr,
            )

        raise RuntimeError(
            f"Rollback alias update failed: {exc}"
        ) from exc

    # Record completion after the alias has changed.
    try:
        client.set_model_version_tag(
            MODEL_NAME,
            target_version,
            "rollback_status",
            "completed",
        )
    except Exception as exc:
        raise RuntimeError(
            "Production alias was changed successfully, but the "
            "rollback completion audit tag could not be recorded. "
            "Verify production before retrying."
        ) from exc

    print(
        f"\nRollback successful: production now points to "
        f"version {target_version}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely roll back the production model."
    )

    parser.add_argument("--target-version", required=True)
    parser.add_argument("--expected-current", required=True)
    parser.add_argument("--reason", required=True)

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve and execute the rollback.",
    )

    args = parser.parse_args()

    try:
        rollback_model(
            args.target_version,
            args.expected_current,
            args.reason,
            args.approve,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
