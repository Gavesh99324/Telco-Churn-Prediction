"""Data preprocessing pipeline for Telco Customer Churn"""
import pickle
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.data_ingestion import DataIngestion
from src.handle_missing_values import MissingValueHandler
from src.outlier_detection import OutlierDetector
from src.feature_engineering import FeatureEngineer
from src.feature_binning import FeatureBinner
from src.feature_encoding import FeatureEncoder
from src.feature_scaling import FeatureScaler
from src.data_splitter import DataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Complete data preprocessing pipeline matching notebooks 02 & 03"""

    def __init__(self,
                 data_path: str = 'data/raw/telco_churn.csv',
                 output_dir: str = 'artifacts/data',
                 save_intermediates: bool = False):
        """
        Initialize DataPipeline

        Args:
            data_path: Path to raw data CSV
            output_dir: Directory to save processed data
            save_intermediates: If True, save intermediate outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.save_intermediates = save_intermediates

        # Initialize all components
        self.ingestion = DataIngestion(data_path=data_path)
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector(multiplier=1.5)
        self.feature_engineer = FeatureEngineer()
        self.binner = FeatureBinner()
        self.encoder = FeatureEncoder()
        self.scaler = FeatureScaler()
        self.splitter = DataSplitter(test_size=0.2, random_state=42)

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Contract and quality gates for reproducible preprocessing.
        self.required_columns = list(self.ingestion.expected_schema.keys())
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.allowed_categories = {
            'gender': {'Male', 'Female'},
            'SeniorCitizen': {'Yes', 'No'},
            'Partner': {'Yes', 'No'},
            'Dependents': {'Yes', 'No'},
            'PhoneService': {'Yes', 'No'},
            'MultipleLines': {'Yes', 'No', 'No phone service'},
            'InternetService': {'DSL', 'Fiber optic', 'No'},
            'OnlineSecurity': {'Yes', 'No', 'No internet service'},
            'OnlineBackup': {'Yes', 'No', 'No internet service'},
            'DeviceProtection': {'Yes', 'No', 'No internet service'},
            'TechSupport': {'Yes', 'No', 'No internet service'},
            'StreamingTV': {'Yes', 'No', 'No internet service'},
            'StreamingMovies': {'Yes', 'No', 'No internet service'},
            'Contract': {'Month-to-month', 'One year', 'Two year'},
            'PaperlessBilling': {'Yes', 'No'},
            'PaymentMethod': {
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)',
                'Bank transfer', 'Credit card'
            },
            'Churn': {'Yes', 'No'}
        }
        self.quality_thresholds = {
            'max_null_ratio': 0.01,
            'max_duplicate_ratio': 0.01,
            'min_minority_class_ratio': 0.10,
        }

    def _save_json_report(self, filename: str, report: dict):
        """Persist JSON report in artifact directory."""
        report_path = Path(self.output_dir) / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("   ✅ Saved %s", report_path)

    def _validate_preprocessing_contract(self, df: pd.DataFrame):
        """Validate strict schema, dtypes, and allowed category contracts."""
        errors = []

        missing_cols = sorted(set(self.required_columns) - set(df.columns))
        extra_cols = sorted(set(df.columns) - set(self.required_columns))

        if missing_cols:
            errors.append(f"missing required columns: {missing_cols}")
        if extra_cols:
            errors.append(f"unexpected columns before encoding: {extra_cols}")

        for col in self.numeric_columns:
            if (
                col in df.columns
                and not pd.api.types.is_numeric_dtype(df[col])
            ):
                errors.append(
                    f"column '{col}' must be numeric, found {df[col].dtype}"
                )

        for col, allowed_values in self.allowed_categories.items():
            if col not in df.columns:
                continue
            observed_values = set(df[col].dropna().astype(str).unique())
            invalid_values = sorted(observed_values - allowed_values)
            if invalid_values:
                errors.append(
                    f"column '{col}' has disallowed values: {invalid_values}"
                )

        if errors:
            error_text = "\n - ".join(errors)
            raise ValueError(
                "Preprocessing schema contract validation failed:\n"
                f" - {error_text}"
            )

    def _build_data_quality_report(
        self,
        df: pd.DataFrame,
        X_scaled: pd.DataFrame,
        y_encoded: np.ndarray
    ) -> dict:
        """Build quality report used by quality gates and lineage artifacts."""
        total_cells = float(df.shape[0] * df.shape[1]) if df.shape[0] else 1.0
        total_missing = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        y_series = pd.Series(y_encoded)
        class_distribution = {
            str(k): int(v)
            for k, v in y_series.value_counts().to_dict().items()
        }
        minority_ratio = (
            float(y_series.value_counts(normalize=True).min())
            if not y_series.empty else 0.0
        )

        return {
            'total_rows': int(df.shape[0]),
            'total_columns': int(df.shape[1]),
            'feature_count_after_encoding': int(X_scaled.shape[1]),
            'total_missing': total_missing,
            'null_ratio': total_missing / total_cells,
            'duplicate_rows': duplicate_rows,
            'duplicate_ratio': duplicate_rows / max(1, len(df)),
            'class_distribution': class_distribution,
            'minority_class_ratio': minority_ratio,
            'thresholds': self.quality_thresholds,
        }

    def _enforce_data_quality_gates(self, report: dict):
        """Fail fast when quality thresholds are violated."""
        failures = []

        if report['null_ratio'] > self.quality_thresholds['max_null_ratio']:
            failures.append(
                "null ratio gate failed: "
                f"{report['null_ratio']:.4f} > "
                f"{self.quality_thresholds['max_null_ratio']:.4f}"
            )

        if (
            report['duplicate_ratio']
            > self.quality_thresholds['max_duplicate_ratio']
        ):
            failures.append(
                "duplicate ratio gate failed: "
                f"{report['duplicate_ratio']:.4f} > "
                f"{self.quality_thresholds['max_duplicate_ratio']:.4f}"
            )

        if (
            report['minority_class_ratio']
            < self.quality_thresholds['min_minority_class_ratio']
        ):
            failures.append(
                "class imbalance gate failed: "
                f"{report['minority_class_ratio']:.4f} < "
                f"{self.quality_thresholds['min_minority_class_ratio']:.4f}"
            )

        if failures:
            failure_text = "\n - ".join(failures)
            raise ValueError(
                "Data quality gates failed:\n"
                f" - {failure_text}"
            )

    def save_artifacts(self, X_train, X_test, y_train, y_test):
        """
        Save all artifacts to disk

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
        """
        logger.info("="*70)
        logger.info("SAVING ARTIFACTS")
        logger.info("="*70)

        # Save data splits
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        pd.DataFrame(y_train, columns=['Churn']).to_csv(
            f'{self.output_dir}/y_train.csv', index=False
        )
        pd.DataFrame(y_test, columns=['Churn']).to_csv(
            f'{self.output_dir}/y_test.csv', index=False
        )
        logger.info("   ✅ Saved train/test splits")

        # Save transformers
        with open(f'{self.output_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoder.label_encoders, f)
        logger.info("   ✅ Saved label encoders")

        with open(f'{self.output_dir}/target_encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder.target_encoder, f)
        logger.info("   ✅ Saved target encoder")

        with open(f'{self.output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler.scaler, f)
        logger.info("   ✅ Saved scaler")

        # Save feature names
        with open(f'{self.output_dir}/feature_names.pkl', 'wb') as f:
            pickle.dump({'feature_names': X_train.columns.tolist()}, f)
        logger.info("   ✅ Saved feature names")

        logger.info("="*70)
        logger.info(f"✅ ALL ARTIFACTS SAVED TO: {self.output_dir}")
        logger.info("="*70)

    def run(self, save_artifacts: bool = True):
        """
        Execute complete data pipeline

        Pipeline stages (matching notebooks 02 & 03):
        1. Data Ingestion (load, validate, convert types)
        2. Missing Value Handling (impute TotalCharges)
        3. Outlier Detection (detect only, no removal)
        4. Feature Engineering (create 9 new features)
        5. Feature Binning (tenure groups)
        6. Feature Encoding (label encode binary, one-hot multiclass)
        7. Feature Scaling (StandardScaler on all features)
        8. Train/Test Split (80/20, stratified)

        Args:
            save_artifacts: If True, save processed data and transformers

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING COMPLETE DATA PIPELINE")
        logger.info("="*70)
        logger.info(f"Data source: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*70 + "\n")

        # Stage 1: Data Ingestion
        df = self.ingestion.ingest()
        if self.save_intermediates:
            df.to_csv(f'{self.output_dir}/01_ingested.csv', index=False)

        # Stage 2: Missing Value Handling
        df = self.missing_handler.handle_missing(df)
        if self.save_intermediates:
            df.to_csv(f'{self.output_dir}/02_missing_handled.csv', index=False)

        # Stage 3: Outlier Detection (detect only, don't remove)
        df = self.outlier_detector.handle_outliers(df, remove=False)

        # Strict contract validation before any artifact persistence.
        self._validate_preprocessing_contract(df)

        # Save cleaned data
        cleaned_path = f'{self.output_dir}/telco_churn_cleaned.csv'
        df.to_csv(cleaned_path, index=False)
        logger.info(f"\n✅ Cleaned data saved to: {cleaned_path}\n")

        # Stage 4: Separate features and target
        logger.info("="*70)
        logger.info("PREPARING FOR FEATURE ENGINEERING")
        logger.info("="*70)

        target_col = 'Churn'
        id_col = 'customerID'

        # Drop customerID and separate target
        if id_col in df.columns:
            df = df.drop(id_col, axis=1)
            logger.info(f"   Dropped {id_col}")

        y = df[target_col].copy()
        X = df.drop(target_col, axis=1)

        logger.info(f"   Features: {X.shape}")
        logger.info(f"   Target: {y.shape}")
        logger.info("="*70 + "\n")

        # Stage 5: Feature Engineering
        X = self.feature_engineer.create_all_features(X)
        if self.save_intermediates:
            X.to_csv(f'{self.output_dir}/03_engineered.csv', index=False)

        # Stage 6: Feature Binning
        X = self.binner.create_all_bins(X)
        if self.save_intermediates:
            X.to_csv(f'{self.output_dir}/04_binned.csv', index=False)

        # Stage 7: Feature Encoding
        X_encoded, y_encoded = self.encoder.fit_transform(X, target_col=None)
        # Encode target separately
        y_encoded = self.encoder.encode_target(y)

        if self.save_intermediates:
            X_encoded.to_csv(f'{self.output_dir}/05_encoded.csv', index=False)

        # Stage 8: Feature Scaling
        X_scaled = self.scaler.fit_transform(X_encoded)
        if self.save_intermediates:
            X_scaled.to_csv(f'{self.output_dir}/06_scaled.csv', index=False)

        # Persist schema contract and quality report for lineage checks.
        schema_contract = {
            'required_input_columns': self.required_columns,
            'numeric_columns': self.numeric_columns,
            'allowed_categories': {
                k: sorted(v) for k, v in self.allowed_categories.items()
            },
            'feature_names': X_scaled.columns.tolist(),
            'feature_dtypes': {
                col: str(dtype) for col, dtype in X_scaled.dtypes.items()
            },
        }
        self._save_json_report('schema_contract.json', schema_contract)

        quality_report = self._build_data_quality_report(
            df,
            X_scaled,
            y_encoded
        )
        self._save_json_report('data_quality_report.json', quality_report)
        self._enforce_data_quality_gates(quality_report)

        # Stage 9: Train/Test Split
        X_train, X_test, y_train, y_test = self.splitter.split(
            X_scaled, y_encoded, stratify=True
        )

        # Save artifacts
        if save_artifacts:
            self.save_artifacts(X_train, X_test, y_train, y_test)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("🎉 DATA PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info("📊 FINAL DATASET SUMMARY:")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        logger.info(f"   Total features: {X_train.shape[1]}")
        # Exclude target
        logger.info(f"   Original features: {df.shape[1] - 1}")
        logger.info(
            "   Engineered features: %s",
            len(self.feature_engineer.created_features)
        )
        logger.info(f"   Label encoded: {len(self.encoder.encoded_columns)}")
        logger.info(f"   One-hot encoded: {len(self.encoder.onehot_columns)}")
        logger.info("="*70 + "\n")

        return X_train, X_test, y_train, y_test


def main():
    """Main execution function"""
    pipeline = DataPipeline(
        data_path='data/raw/telco_churn.csv',
        output_dir='artifacts/data',
        save_intermediates=False
    )

    X_train, X_test, _, _ = pipeline.run(save_artifacts=True)

    print("\n✅ Pipeline execution complete!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")


if __name__ == '__main__':
    main()
