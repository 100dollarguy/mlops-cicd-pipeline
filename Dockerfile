FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5002

# MLFLOW_TRACKING_URI must point to a reachable MLflow tracking server.
# The production alias is resolved when the container starts.
CMD ["mlflow", "models", "serve", "--model-uri", "models:/BreastCancerClassifier@production", "--host", "0.0.0.0", "--port", "5002", "--env-manager", "local"]
