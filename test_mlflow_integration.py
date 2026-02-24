"""Test script to verify MLflow integration"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.mlflow_utils import MLflowManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mlflow_connection():
    """Test MLflow connection and basic operations"""
    print("\n" + "="*70)
    print("TESTING MLFLOW INTEGRATION")
    print("="*70)
    
    try:
        # Initialize MLflow Manager
        print("\n1. Initializing MLflow Manager...")
        mlflow_manager = MLflowManager(
            tracking_uri='http://localhost:5000',
            experiment_name='test-experiment',
            enable_autolog=False
        )
        print("✅ MLflow Manager initialized successfully")
        
        # Test run creation and logging
        print("\n2. Creating test run and logging data...")
        with mlflow_manager.start_run(run_name="test_run"):
            # Log parameters
            mlflow_manager.log_params({
                'test_param_1': 'value1',
                'test_param_2': 42,
                'test_param_3': 3.14
            })
            print("✅ Logged parameters")
            
            # Log metrics
            mlflow_manager.log_metrics({
                'test_accuracy': 0.85,
                'test_f1': 0.82,
                'test_precision': 0.88
            })
            print("✅ Logged metrics")
            
            # Log tags
            mlflow_manager.set_tags({
                'test_tag': 'integration_test',
                'status': 'testing'
            })
            print("✅ Set tags")
        
        # Get experiment info
        print("\n3. Retrieving experiment information...")
        exp_info = mlflow_manager.get_experiment_info()
        print(f"✅ Experiment: {exp_info['name']}")
        print(f"   ID: {exp_info['experiment_id']}")
        print(f"   Artifact Location: {exp_info['artifact_location']}")
        
        # Search runs
        print("\n4. Searching runs...")
        runs = mlflow_manager.search_runs(max_results=5)
        print(f"✅ Found {len(runs)} run(s)")
        
        print("\n" + "="*70)
        print("✅ ALL MLFLOW TESTS PASSED!")
        print("="*70)
        print("\n📌 MLflow UI: http://localhost:5000")
        print("📌 You can view experiments and runs in the browser")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ MLFLOW TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        print("\n📌 Make sure MLflow server is running:")
        print("   Option 1: docker-compose up mlflow")
        print("   Option 2: mlflow server --host 0.0.0.0 --port 5000")
        return False


if __name__ == '__main__':
    success = test_mlflow_connection()
    sys.exit(0 if success else 1)
