#!/bin/bash
# MLflow entrypoint
set -e

echo "Starting MLflow tracking server..."
mlflow server --host 0.0.0.0 --port 5000
