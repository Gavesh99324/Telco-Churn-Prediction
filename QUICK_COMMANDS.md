# Quick Command Reference - Data Pipeline

## 🚀 Run Pipeline

```bash
# Full pipeline execution
conda activate telco-churn
cd "g:\Python Codes\Telco Churn Prediction"
python -m pipelines.data_pipeline
```

## 🧪 Run Tests

```bash
# All unit tests
python -m pytest tests/unit/ -v

# Specific test files
python -m pytest tests/unit/test_data_ingestion.py -v
python -m pytest tests/unit/test_missing_values.py -v
python -m pytest tests/unit/test_feature_encoding.py -v
python -m pytest tests/unit/test_pipeline.py -v

# With coverage report
python -m pytest tests/unit/ --cov=src --cov=pipelines --cov-report=html
open htmlcov/index.html
```

## 📊 Verify Artifacts

```bash
# Check generated files
ls artifacts/data/

# Should see:
# - X_train.csv, X_test.csv
# - y_train.csv, y_test.csv
# - telco_churn_cleaned.csv
# - label_encoders.pkl, target_encoder.pkl
# - scaler.pkl, feature_names.pkl
```

## 🐍 Python Usage Examples

### Run Complete Pipeline
```python
from pipelines.data_pipeline import DataPipeline

pipeline = DataPipeline(
    data_path='data/raw/telco_churn.csv',
    output_dir='artifacts/data'
)
X_train, X_test, y_train, y_test = pipeline.run()
print(f"Training: {X_train.shape}, Test: {X_test.shape}")
```

### Use Individual Modules
```python
# Data Ingestion
from src.data_ingestion import DataIngestion
ingestion = DataIngestion()
df = ingestion.ingest()

# Missing Values
from src.handle_missing_values import MissingValueHandler
handler = MissingValueHandler()
df = handler.handle_missing(df)

# Feature Engineering
from src.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
df = engineer.create_all_features(df)

# Encoding
from src.feature_encoding import FeatureEncoder
encoder = FeatureEncoder()
X_encoded = encoder.fit_transform(df, target_col='Churn')

# Scaling
from src.feature_scaling import FeatureScaler
scaler = FeatureScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split
from src.data_splitter import DataSplitter
splitter = DataSplitter(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X_scaled, y, stratify=True)
```

## 🔍 Verify Against Notebook

```python
import pandas as pd

# Load existing notebook artifacts
X_train_nb = pd.read_csv('artifacts/data/X_train.csv')
X_test_nb = pd.read_csv('artifacts/data/X_test.csv')

print(f"Notebook - Train: {X_train_nb.shape}, Test: {X_test_nb.shape}")
print(f"Features: {X_train_nb.columns.tolist()[:5]}...")

# Now run pipeline and compare
from pipelines.data_pipeline import DataPipeline
pipeline = DataPipeline()
X_train_pipe, X_test_pipe, _, _ = pipeline.run()

print(f"Pipeline - Train: {X_train_pipe.shape}, Test: {X_test_pipe.shape}")
print(f"Shapes match: {X_train_nb.shape == X_train_pipe.shape}")
```

## 📝 Check Pipeline Status

```python
# Check what artifacts exist
import os
artifacts = os.listdir('artifacts/data')
print(f"Found {len(artifacts)} artifacts:")
for f in sorted(artifacts):
    print(f"  - {f}")

# Load and inspect
import pandas as pd
X_train = pd.read_csv('artifacts/data/X_train.csv')
print(f"\nX_train shape: {X_train.shape}")
print(f"Features: {X_train.shape[1]}")
print(f"Samples: {X_train.shape[0]}")
print(f"Missing values: {X_train.isnull().sum().sum()}")
```

## 🛠️ Debugging

```bash
# Run with Python debugger
python -m pdb pipelines/data_pipeline.py

# Run single test with output
python -m pytest tests/unit/test_data_ingestion.py::test_complete_ingestion_pipeline -v -s

# Check import errors
python -c "from src.data_ingestion import DataIngestion; print('OK')"
python -c "from pipelines.data_pipeline import DataPipeline; print('OK')"
```

## 📈 Performance Check

```python
import time
from pipelines.data_pipeline import DataPipeline

start = time.time()
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run()
elapsed = time.time() - start

print(f"Pipeline execution time: {elapsed:.2f} seconds")
print(f"Processed {len(X_train) + len(X_test)} samples")
print(f"Speed: {(len(X_train) + len(X_test)) / elapsed:.0f} samples/sec")
```

## 🔄 Clean and Rebuild

```bash
# Remove old artifacts
rm -rf artifacts/data/*

# Run pipeline fresh
python -m pipelines.data_pipeline

# Verify new artifacts
ls -lh artifacts/data/
```

---
**Quick Access:** Keep this file open for easy copy-paste!
