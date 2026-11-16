# MLOps CI/CD Pipeline

An end-to-end machine learning pipeline demonstrating reproducible data and model workflows, experiment tracking, model lifecycle management, automated testing, and continuous integration.

The project uses the Breast Cancer Wisconsin dataset and a Logistic Regression classifier to demonstrate core MLOps practices, including model validation, controlled promotion, and rollback using MLflow Model Registry.

## Project Overview

This repository brings together:

- **DVC** for dataset and pipeline versioning.
- **MLflow** for experiment tracking, parameter logging, metrics, model artifacts, and model registry.
- **Scikit-learn** for data processing, model training, and evaluation.
- **Pytest** for automated unit and integration testing.
- **GitHub Actions** for continuous integration.
- **Python lifecycle scripts** for model validation, production promotion, and rollback.

## Architecture

```text
                 ┌────────────────────────┐
                 │ Breast Cancer Wisconsin│
                 │ Dataset                │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Data Preparation       │
                 │ prepare_data.py        │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ DVC Data & Pipeline    │
                 │ Versioning             │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Model Training         │
                 │ Logistic Regression    │
                 └───────────┬────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
        ┌──────────────────┐  ┌────────────────────┐
        │ Model Evaluation │  │ MLflow Tracking    │
        │ Metrics/Reports  │  │ Parameters/Metrics │
        └──────────────────┘  │ Model Artifacts    │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ MLflow Model       │
                              │ Registry           │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Model Validation   │
                              │ Accuracy >= 0.95   │
                              │ F1-score >= 0.95   │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Staging Alias      │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Controlled         │
                              │ Production         │
                              │ Promotion          │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Production Alias   │
                              └────────────────────┘

                 ┌────────────────────────┐
                 │ Rollback Workflow      │
                 │ Validation + Audit     │
                 │ Metadata + Alias       │
                 │ Update                 │
                 └────────────────────────┘

                 ┌────────────────────────┐
                 │ GitHub Actions         │
                 │ Automated Tests + CI   │
                 └────────────────────────┘
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Data & Pipeline Versioning | DVC |
| Testing | Pytest, pytest-cov |
| CI | GitHub Actions |
| Environment Management | Conda |
| Metadata Storage | SQLite |
| Model Artifact Storage | Local filesystem |

## Repository Structure

```text
mlops-cicd-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw.dvc
│   ├── raw/
│   └── processed/
├── mlflow/
│   ├── database/
│   └── artifacts/
├── reports/
├── scripts/
│   ├── __init__.py
│   ├── model_utils.py
│   ├── promote_model.py
│   └── rollback_model.py
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_mlflow.py
│   ├── test_model_lifecycle.py
│   ├── test_pipeline.py
│   ├── test_prepare_data.py
│   └── test_training.py
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── environment.yml
├── pytest.ini
└── requirements.txt
```

Generated data, MLflow runtime files, and other local artifacts are excluded from Git where appropriate.

## Getting Started

### Prerequisites

- Python 3.12
- Conda
- Git

### 1. Clone the repository

```bash
git clone https://github.com/100dollarguy/mlops-cicd-pipeline.git
cd mlops-cicd-pipeline
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate mlops-cicd
```

Alternatively, install dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python src/prepare_data.py
```

The preparation script uses Scikit-learn's Breast Cancer Wisconsin dataset and creates stratified training and testing CSV files.

### 4. Start the MLflow tracking server

In a separate terminal, activate the environment and run:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5001 \
  --backend-store-uri sqlite:///mlflow/database/mlflow.db \
  --artifacts-destination ./mlflow/artifacts
```

Configure the tracking URI in the project terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Open the MLflow UI:

http://127.0.0.1:5001

The tracking server uses a local SQLite backend and local artifact storage in the development setup.

### 5. Run the DVC pipeline

```bash
dvc repro
```

DVC executes the pipeline stages when their dependencies or outputs have changed.

The training process logs parameters, evaluation metrics, reports, and the trained model to MLflow.

To inspect the pipeline:

```bash
dvc dag
```

To inspect tracked metrics:

```bash
dvc metrics show
```

## Model Evaluation

The training pipeline uses:

- StandardScaler for feature scaling.
- Logistic Regression for classification.
- A stratified 80/20 training/testing split.
- A fixed random seed for reproducibility.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Metrics and classification reports are written to the `reports/` directory.

## Experiment Tracking with MLflow

Each training run records:

- Model hyperparameters and configuration.
- Evaluation metrics.
- Metrics and classification report artifacts.
- Trained Scikit-learn model artifact.
- Model input/output signature.

MLflow provides experiment tracking and stores model artifacts for subsequent registration and lifecycle operations.

## Model Registry and Lifecycle Management

The project uses MLflow Model Registry to manage registered model versions and their lifecycle.

The registered model is:

`BreastCancerClassifier`

### Model validation

Before a model can be promoted, it must satisfy the following conditions:

| Validation | Requirement |
|---|---:|
| Model status | READY |
| Validation status | passed |
| Minimum accuracy | 0.95 |
| Minimum F1-score | 0.95 |
| Metric values | Finite and between 0 and 1 |

The validation logic is implemented in:

```text
scripts/model_utils.py
```

Models that fail validation are rejected before promotion or rollback.

### Model aliases

The registry uses two aliases:

| Alias | Purpose |
|---|---|
| `staging` | Identifies the candidate model version |
| `production` | Identifies the currently approved production model version |

Aliases provide a way to reference model versions without hardcoding version numbers throughout the workflow.

### Production promotion

The promotion script is:

```text
scripts/promote_model.py
```

Promotion includes:

1. Verify that the requested model version matches the staging alias.
2. Validate the model status and validation metrics.
3. Check the current production version.
4. Verify the expected current production version.
5. Display candidate information and promotion details.
6. Require explicit approval before changing the production alias.

**Dry run:**

```bash
python -m scripts.promote_model \
  --version VERSION \
  --expected-current CURRENT_PRODUCTION_VERSION
```

**Execute promotion:**

```bash
python -m scripts.promote_model \
  --version VERSION \
  --expected-current CURRENT_PRODUCTION_VERSION \
  --approve
```

Replace the version placeholders with actual MLflow model version numbers.

The promotion script defaults to dry-run mode. The `--approve` flag is required to change the production alias.

### Model rollback

The rollback script is:

```text
scripts/rollback_model.py
```

Rollback requires:

- A target model version.
- The expected current production version.
- A non-empty rollback reason.
- A model that passes validation.
- Explicit approval before changing production.

**Dry run:**

```bash
python -m scripts.rollback_model \
  --target-version TARGET_VERSION \
  --expected-current CURRENT_PRODUCTION_VERSION \
  --reason "Rollback due to model performance regression"
```

**Execute rollback:**

```bash
python -m scripts.rollback_model \
  --target-version TARGET_VERSION \
  --expected-current CURRENT_PRODUCTION_VERSION \
  --reason "Rollback due to model performance regression" \
  --approve
```

The rollback workflow records audit metadata on the target model version, including:

- `rollback_reason`
- `rollback_timestamp`
- `rollback_from_version`
- `rollback_status`
- `rollback_error` when an alias update fails

The rollback status tracks the operation through:

- `pending`
- `completed`
- `failed`

The workflow records rollback intent before changing the production alias and attempts to record failures if the alias update does not succeed.

**Important:** MLflow alias changes and audit-tag updates are separate operations, not a single atomic transaction. The expected-current check reduces the risk of overwriting an unexpected production version but does not provide atomic compare-and-swap protection.

The production alias is a registry pointer. This project currently implements model lifecycle management through MLflow aliases; it does not yet implement a separate model-serving or deployment environment.

## Testing

Run the complete test suite:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest -v \
  --cov=src \
  --cov=scripts \
  --cov-report=term-missing \
  --cov-fail-under=90
```

The test suite covers:

- Data loading and validation.
- Dataset preparation.
- Model training and evaluation.
- MLflow experiment integration.
- Model version validation.
- Promotion approval and rejection scenarios.
- Rollback validation and audit handling.
- CLI behavior and error handling.
- Pipeline report generation.

Latest verified test results:

| Metric | Result |
|---|---:|
| Tests collected | 45 |
| Tests passed | 45 |
| Tests failed | 0 |
| Total code coverage | 96.33% |
| Required coverage | 90% |

## Continuous Integration

GitHub Actions runs the CI workflow on pushes and pull requests targeting `master`.

The workflow:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs project dependencies.
4. Generates the dataset.
5. Validates the DVC pipeline structure.
6. Runs automated tests with coverage reporting.
7. Enforces a minimum 90% coverage requirement.

The hosted CI workflow generates the dataset directly rather than accessing the developer's machine-specific local DVC remote.

MLflow integration is disabled in the CI environment where a running local tracking server is not available.

## DVC Storage

The development setup uses a local DVC remote for storing dataset versions and cached artifacts.

The local remote configuration is intentionally kept outside the shared Git repository. Each developer must configure an accessible DVC remote for their own environment.

## Current Scope

Implemented:

- Reproducible dataset preparation.
- DVC dataset and pipeline tracking.
- MLflow experiment tracking.
- Model training and evaluation.
- MLflow model registration and version management.
- Model validation with configurable quality thresholds.
- Staging and production model aliases.
- Controlled production promotion with explicit approval.
- Production version checks using expected-current arguments.
- Rollback with audit metadata and failure handling.
- Automated unit and integration tests.
- GitHub Actions continuous integration.
- Code coverage enforcement.

Future improvements:

- Dedicated staging and production deployment environments.
- Automated model serving and inference endpoints.
- Deployment health checks and monitoring.
- Automated model performance and data drift detection.
- Remote DVC storage for shared environments.
- Automated deployment and rollback integration.

---

**Author:** [ROHAN D R](https://github.com/100dollarguy)
