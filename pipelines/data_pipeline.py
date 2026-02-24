"""Data preprocessing pipeline for Telco Customer Churn"""
import os
import pickle
import pandas as pd
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
            customer_ids = df[id_col].copy()
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
        logger.info(f"📊 FINAL DATASET SUMMARY:")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        logger.info(f"   Total features: {X_train.shape[1]}")
        # Exclude target
        logger.info(f"   Original features: {df.shape[1] - 1}")
        logger.info(
            f"   Engineered features: {len(self.feature_engineer.created_features)}")
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

    X_train, X_test, y_train, y_test = pipeline.run(save_artifacts=True)

    print("\n✅ Pipeline execution complete!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")


if __name__ == '__main__':
    main()
