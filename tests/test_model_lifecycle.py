from types import SimpleNamespace

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode

from scripts import model_utils, promote_model, rollback_model


# --------------------------------------------------
# Shared test helpers
# --------------------------------------------------

def make_model(
    version="6",
    status="READY",
    tags=None,
):
    return SimpleNamespace(
        version=version,
        status=status,
        tags=tags if tags is not None else {
            "validation_status": "passed",
            "validation_accuracy": "0.9825",
            "validation_f1_score": "0.9861",
        },
    )


class FakeValidationClient:
    def __init__(self, model):
        self.model = model

    def get_model_version(self, name, version):
        assert name == model_utils.MODEL_NAME
        return self.model


class FakePromotionClient:
    def __init__(
        self,
        staging_version="6",
        production_version="5",
        candidate=None,
    ):
        self.staging_version = staging_version
        self.production_version = production_version
        self.candidate = candidate or make_model()
        self.alias_updates = []

    def get_model_version_by_alias(self, name, alias):
        if alias == "staging":
            return SimpleNamespace(version=self.staging_version)

        if alias == "production":
            if self.production_version is None:
                raise MlflowException(
                    "Alias not found",
                    error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
                )

            return SimpleNamespace(version=self.production_version)

        raise AssertionError(f"Unexpected alias: {alias}")

    def get_model_version(self, name, version):
        return self.candidate

    def set_registered_model_alias(self, name, alias, version):
        self.alias_updates.append((name, alias, version))


class FakeRollbackClient:
    def __init__(
        self,
        current_version="6",
        target=None,
        alias_error=None,
    ):
        self.current_version = current_version
        self.target = target or make_model(version="5")
        self.alias_updates = []
        self.tags = []
        self.alias_error = alias_error
        self.tag_error = None

    def get_model_version_by_alias(self, name, alias):
        assert alias == "production"
        return SimpleNamespace(version=self.current_version)

    def get_model_version(self, name, version):
        return self.target

    def set_registered_model_alias(self, name, alias, version):
        if self.alias_error:
            raise self.alias_error

        self.alias_updates.append((name, alias, version))
        self.current_version = version

    def set_model_version_tag(self, name, version, key, value):
        if self.tag_error:
            raise self.tag_error

        self.tags.append((name, version, key, value))


# --------------------------------------------------
# Shared MLflow client tests
# --------------------------------------------------

def test_get_mlflow_client_requires_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        model_utils.get_mlflow_client()


def test_get_mlflow_client_uses_configured_uri(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001",
    )

    class FakeMlflowClient:
        def __init__(self, tracking_uri):
            self.tracking_uri = tracking_uri

    monkeypatch.setattr(
        model_utils,
        "MlflowClient",
        FakeMlflowClient,
    )

    client = model_utils.get_mlflow_client()

    assert client.tracking_uri == "http://127.0.0.1:5001"


# --------------------------------------------------
# Model validation tests
# --------------------------------------------------

def test_validate_model_version_passes():
    model = make_model()
    client = FakeValidationClient(model)

    result, accuracy, f1_score = model_utils.validate_model_version(
        client, "6"
    )

    assert result is model
    assert accuracy == 0.9825
    assert f1_score == 0.9861


def test_validate_model_version_rejects_non_ready_model():
    client = FakeValidationClient(
        make_model(status="PENDING_REGISTRATION")
    )

    with pytest.raises(ValueError, match="not READY"):
        model_utils.validate_model_version(client, "6")


def test_validate_model_version_rejects_missing_validation_status():
    client = FakeValidationClient(
        make_model(tags={"validation_accuracy": "0.99"})
    )

    with pytest.raises(ValueError, match="not passed"):
        model_utils.validate_model_version(client, "6")


@pytest.mark.parametrize(
    "tags",
    [
        {
            "validation_status": "passed",
            "validation_accuracy": "0.99",
        },
        {
            "validation_status": "passed",
            "validation_accuracy": "invalid",
            "validation_f1_score": "0.99",
        },
    ],
)
def test_validate_model_version_rejects_missing_or_invalid_metrics(tags):
    client = FakeValidationClient(make_model(tags=tags))

    with pytest.raises(ValueError, match="missing or invalid"):
        model_utils.validate_model_version(client, "6")


@pytest.mark.parametrize(
    "metric,value",
    [
        ("validation_accuracy", "0.94"),
        ("validation_f1_score", "0.94"),
    ],
)
def test_validate_model_version_rejects_low_metrics(metric, value):
    tags = make_model().tags
    tags[metric] = value

    client = FakeValidationClient(make_model(tags=tags))

    with pytest.raises(ValueError, match="below"):
        model_utils.validate_model_version(client, "6")


@pytest.mark.parametrize(
    "metric,value",
    [
        ("validation_accuracy", "nan"),
        ("validation_f1_score", "inf"),
        ("validation_accuracy", "-inf"),
    ],
)
def test_validate_model_version_rejects_non_finite_metrics(metric, value):
    tags = make_model().tags
    tags[metric] = value

    client = FakeValidationClient(make_model(tags=tags))

    with pytest.raises(ValueError, match="finite"):
        model_utils.validate_model_version(client, "6")


@pytest.mark.parametrize(
    "metric,value",
    [
        ("validation_accuracy", "1.01"),
        ("validation_f1_score", "-0.01"),
    ],
)
def test_validate_model_version_rejects_out_of_range_metrics(metric, value):
    tags = make_model().tags
    tags[metric] = value

    client = FakeValidationClient(make_model(tags=tags))

    with pytest.raises(ValueError, match="between 0 and 1"):
        model_utils.validate_model_version(client, "6")


# --------------------------------------------------
# Promotion tests
# --------------------------------------------------

def test_promotion_dry_run_does_not_change_alias(monkeypatch, capsys):
    client = FakePromotionClient()

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    promote_model.promote_model("6", False, "5")

    output = capsys.readouterr().out

    assert "DRY RUN" in output
    assert client.alias_updates == []


def test_promotion_rejects_wrong_staging_version(monkeypatch):
    client = FakePromotionClient(staging_version="5")

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="not the current staging"):
        promote_model.promote_model("6", False, None)


def test_promotion_rejects_stale_production_version(monkeypatch):
    client = FakePromotionClient(production_version="4")

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        promote_model.promote_model("6", False, "5")


def test_promotion_updates_alias_only_when_approved(monkeypatch):
    client = FakePromotionClient()

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    promote_model.promote_model("6", True, "5")

    assert client.alias_updates == [
        ("BreastCancerClassifier", "production", "6")
    ]


def test_promotion_rejects_already_production_version(monkeypatch):
    client = FakePromotionClient(production_version="6")

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="already in production"):
        promote_model.promote_model("6", False, "6")


def test_promotion_rejects_invalid_candidate(monkeypatch):
    invalid_model = make_model(status="FAILED")
    client = FakePromotionClient(candidate=invalid_model)

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="not READY"):
        promote_model.promote_model("6", False, None)


# --------------------------------------------------
# Rollback tests
# --------------------------------------------------

def test_rollback_dry_run_does_not_change_alias(monkeypatch, capsys):
    client = FakeRollbackClient()

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    rollback_model.rollback_model(
        "5",
        "6",
        "Testing rollback dry-run",
        False,
    )

    output = capsys.readouterr().out

    assert "DRY RUN" in output
    assert client.alias_updates == []
    assert client.tags == []


def test_rollback_rejects_wrong_expected_current(monkeypatch):
    client = FakeRollbackClient(current_version="6")

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        rollback_model.rollback_model(
            "5",
            "4",
            "Testing stale version protection",
            False,
        )


def test_rollback_rejects_same_target_and_current(monkeypatch):
    client = FakeRollbackClient()

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="already the production"):
        rollback_model.rollback_model(
            "6",
            "6",
            "Testing identical versions",
            False,
        )


def test_rollback_rejects_empty_reason(monkeypatch):
    client = FakeRollbackClient()

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        rollback_model.rollback_model(
            "5",
            "6",
            "   ",
            False,
        )


def test_rollback_rejects_invalid_target(monkeypatch):
    invalid_model = make_model(
        version="5",
        tags={"validation_status": "failed"},
    )
    client = FakeRollbackClient(target=invalid_model)

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(ValueError, match="not passed"):
        rollback_model.rollback_model(
            "5",
            "6",
            "Testing invalid target",
            False,
        )


def test_rollback_updates_alias_and_audit_tags_when_approved(monkeypatch):
    client = FakeRollbackClient()

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    rollback_model.rollback_model(
        "5",
        "6",
        "Approved rollback test",
        True,
    )

    assert client.alias_updates == [
        ("BreastCancerClassifier", "production", "5")
    ]

    tag_names = {tag[2] for tag in client.tags}

    assert tag_names == {
        "rollback_reason",
        "rollback_timestamp",
        "rollback_from_version",
        "rollback_status",
    }

    assert (
        "BreastCancerClassifier",
        "5",
        "rollback_status",
        "completed",
    ) in client.tags


# --------------------------------------------------
# Additional promotion tests
# --------------------------------------------------

def test_promotion_handles_missing_production_alias(monkeypatch, capsys):
    client = FakePromotionClient(production_version=None)

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    promote_model.promote_model("6", False, None)

    output = capsys.readouterr().out

    assert "Previous production version: None" in output
    assert "DRY RUN" in output
    assert client.alias_updates == []


def test_promotion_rejects_unexpected_mlflow_error(monkeypatch):
    class BrokenClient(FakePromotionClient):
        def get_model_version_by_alias(self, name, alias):
            if alias == "production":
                raise MlflowException("Unexpected backend failure")
            return super().get_model_version_by_alias(name, alias)

    client = BrokenClient()

    monkeypatch.setattr(
        promote_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(MlflowException, match="Unexpected backend failure"):
        promote_model.promote_model("6", False, None)


# --------------------------------------------------
# Additional rollback failure tests
# --------------------------------------------------

def test_rollback_aborts_when_initial_audit_fails(monkeypatch):
    client = FakeRollbackClient()
    client.tag_error = RuntimeError("Audit storage unavailable")

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="unable to record rollback intent"):
        rollback_model.rollback_model(
            "5",
            "6",
            "Testing audit failure",
            True,
        )

    assert client.alias_updates == []


def test_rollback_marks_failed_when_alias_update_fails(monkeypatch):
    client = FakeRollbackClient(
        alias_error=RuntimeError("Alias storage unavailable")
    )

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="alias update failed"):
        rollback_model.rollback_model(
            "5",
            "6",
            "Testing alias failure",
            True,
        )

    assert client.alias_updates == []
    assert (
        "BreastCancerClassifier",
        "5",
        "rollback_status",
        "failed",
    ) in client.tags


def test_rollback_reports_completion_audit_failure(monkeypatch):
    class CompletionAuditFailureClient(FakeRollbackClient):
        def set_model_version_tag(self, name, version, key, value):
            if key == "rollback_status" and value == "completed":
                raise RuntimeError("Completion audit unavailable")

            self.tags.append((name, version, key, value))

    client = CompletionAuditFailureClient()

    monkeypatch.setattr(
        rollback_model,
        "get_mlflow_client",
        lambda: client,
    )

    with pytest.raises(
        RuntimeError,
        match="alias was changed successfully",
    ):
        rollback_model.rollback_model(
            "5",
            "6",
            "Testing completion audit failure",
            True,
        )

    assert client.alias_updates == [
        ("BreastCancerClassifier", "production", "5")
    ]


# --------------------------------------------------
# Promotion CLI tests
# --------------------------------------------------

def test_promotion_cli_calls_promotion_function(monkeypatch):
    captured = {}

    def fake_promote(version, approve, expected_current):
        captured["version"] = version
        captured["approve"] = approve
        captured["expected_current"] = expected_current

    monkeypatch.setattr(
        promote_model,
        "promote_model",
        fake_promote,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "promote_model.py",
            "--version", "6",
            "--expected-current", "5",
            "--approve",
        ],
    )

    promote_model.main()

    assert captured == {
        "version": "6",
        "approve": True,
        "expected_current": "5",
    }


def test_promotion_cli_handles_errors(monkeypatch):
    def fake_promote(version, approve, expected_current):
        raise ValueError("Simulated promotion failure")

    monkeypatch.setattr(
        promote_model,
        "promote_model",
        fake_promote,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "promote_model.py",
            "--version",
            "6",
            "--expected-current",
            "5",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        promote_model.main()

    assert exc.value.code == 1


# --------------------------------------------------
# Rollback CLI tests
# --------------------------------------------------

def test_rollback_cli_calls_rollback_function(monkeypatch):
    captured = {}

    def fake_rollback(
        target_version,
        expected_current,
        reason,
        approve,
    ):
        captured["target_version"] = target_version
        captured["expected_current"] = expected_current
        captured["reason"] = reason
        captured["approve"] = approve

    monkeypatch.setattr(
        rollback_model,
        "rollback_model",
        fake_rollback,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rollback_model.py",
            "--target-version", "5",
            "--expected-current", "6",
            "--reason", "CLI test",
            "--approve",
        ],
    )

    rollback_model.main()

    assert captured == {
        "target_version": "5",
        "expected_current": "6",
        "reason": "CLI test",
        "approve": True,
    }


def test_rollback_cli_handles_errors(monkeypatch):
    def fake_rollback(*args, **kwargs):
        raise ValueError("Simulated rollback failure")

    monkeypatch.setattr(
        rollback_model,
        "rollback_model",
        fake_rollback,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rollback_model.py",
            "--target-version", "5",
            "--expected-current", "6",
            "--reason", "CLI test",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        rollback_model.main()

    assert exc.value.code == 1
