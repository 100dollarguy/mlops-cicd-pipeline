# MLOps CI/CD Pipeline

An end-to-end machine learning pipeline demonstrating reproducible data and model workflows, experiment tracking, automated testing, and continuous integration.

The project uses the Breast Cancer Wisconsin dataset and a Logistic Regression classifier to demonstrate core MLOps practices.

## Project Overview

This repository brings together:

- **DVC** for dataset and pipeline versioning.
- **MLflow** for experiment tracking, parameter logging, metrics, and model artifacts.
- **Scikit-learn** for data processing, model training, and evaluation.
- **Pytest** for automated unit and integration testing.
- **GitHub Actions** for continuous integration.

## Architecture

```text
                 ┌──────────────────────┐
                 │  Breast Cancer       │
                 │  Wisconsin Dataset   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Data Preparation    │
                 │  prepare_data.py     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  DVC Data Versioning │
                 │  data/raw.dvc        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Model Training      │
                 │  train.py            │
                 │  Logistic Regression │
                 └──────────┬───────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
       ┌─────────────────┐     ┌──────────────────┐
       │ Model Evaluation│     │ MLflow Tracking  │
       │ Metrics/Reports │     │ Params/Metrics   │
       └─────────────────┘     │ Model Artifacts  │
                               └──────────────────┘

                 ┌──────────────────────┐
                 │ GitHub Actions       │
                 │ Tests + CI checks    │
                 └──────────────────────┘
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | MLflow |
| Data & Pipeline Versioning | DVC |
| Testing | Pytest, pytest-cov |
| CI | GitHub Actions |
| Environment Management | Conda |

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
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_mlflow.py
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
  --port 5000 \
  --backend-store-uri sqlite:///mlflow/database/mlflow.db \
  --artifacts-destination ./mlflow/artifacts
```

Open the MLflow UI:

http://127.0.0.1:5000

### 5. Run the DVC pipeline

In the project terminal:

```bash
dvc repro
```

DVC executes the training stage when its dependencies or outputs have changed.

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

MLflow uses a local SQLite backend and local artifact storage in the development setup.

## Testing

Run the complete test suite:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=90
```

The test suite covers data validation, dataset preparation, model training, evaluation, report generation, and MLflow integration.

## Continuous Integration

GitHub Actions runs the CI workflow on pushes and pull requests targeting `master` or `main`.

The workflow:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs project dependencies.
4. Generates the dataset.
5. Validates the DVC pipeline structure.
6. Runs automated tests with a minimum 90% coverage requirement.

The hosted CI workflow generates the dataset directly rather than accessing the developer's machine-specific local DVC remote.

## DVC Storage

The development setup uses a local DVC remote for storing dataset versions and cached artifacts.

The local remote configuration is intentionally kept outside the shared Git repository. Each developer must configure an accessible DVC remote for their own environment.

## Current Scope

Implemented:

- Reproducible dataset preparation.
- DVC dataset and pipeline tracking.
- MLflow experiment tracking.
- Model training and evaluation.
- Automated unit and integration tests.
- GitHub Actions continuous integration.

Future improvements:

- Dedicated staging and production environments.
- Controlled model registration and promotion.
- Git-tag-based model promotion workflow.
- Remote DVC storage for shared environments.
- Deployment and model serving automation.

---

**Author:** [ROHAN D R](https://github.com/100dollarguy)
