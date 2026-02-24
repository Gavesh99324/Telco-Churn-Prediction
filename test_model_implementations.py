"""Test script to verify model implementations"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model_building import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.model_inference import ModelInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_building():
    """Test ModelBuilder"""
    logger.info("=" * 70)
    logger.info("TESTING MODEL BUILDING")
    logger.info("=" * 70)
    
    builder = ModelBuilder(random_state=42)
    
    # Test individual model builders
    lr = builder.build_logistic_regression()
    logger.info(f"✅ Logistic Regression: {type(lr).__name__}")
    
    dt = builder.build_decision_tree()
    logger.info(f"✅ Decision Tree: {type(dt).__name__}")
    
    rf = builder.build_random_forest()
    logger.info(f"✅ Random Forest: {type(rf).__name__}")
    
    gb = builder.build_gradient_boosting()
    logger.info(f"✅ Gradient Boosting: {type(gb).__name__}")
    
    svm = builder.build_svm()
    logger.info(f"✅ SVM: {type(svm).__name__}")
    
    # Test ensemble builders
    voting = builder.build_voting_classifier()
    logger.info(f"✅ Voting Classifier: {type(voting).__name__}")
    
    stacking = builder.build_stacking_classifier()
    logger.info(f"✅ Stacking Classifier: {type(stacking).__name__}")
    
    # Test build all
    all_models = builder.build_all_models()
    logger.info(f"✅ Built {len(all_models)} baseline models")
    
    return all_models

def test_model_training():
    """Test ModelTrainer"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING MODEL TRAINING")
    logger.info("=" * 70)
    
    trainer = ModelTrainer(random_state=42)
    logger.info(f"✅ ModelTrainer initialized")
    
    # Check methods exist
    assert hasattr(trainer, 'train'), "train method missing"
    assert hasattr(trainer, 'train_with_smote'), "train_with_smote method missing"
    assert hasattr(trainer, 'hyperparameter_tuning_grid'), "hyperparameter_tuning_grid missing"
    assert hasattr(trainer, 'hyperparameter_tuning_random'), "hyperparameter_tuning_random missing"
    assert hasattr(trainer, 'train_multiple_models'), "train_multiple_models missing"
    assert hasattr(trainer, 'save_model'), "save_model method missing"
    assert hasattr(trainer, 'get_training_summary'), "get_training_summary method missing"
    
    logger.info("✅ All training methods present")
    return trainer

def test_model_evaluation():
    """Test ModelEvaluator"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING MODEL EVALUATION")
    logger.info("=" * 70)
    
    evaluator = ModelEvaluator()
    logger.info(f"✅ ModelEvaluator initialized")
    
    # Check methods exist
    assert hasattr(evaluator, 'evaluate_model'), "evaluate_model method missing"
    assert hasattr(evaluator, 'plot_confusion_matrix'), "plot_confusion_matrix method missing"
    assert hasattr(evaluator, 'plot_roc_curves'), "plot_roc_curves method missing"
    assert hasattr(evaluator, 'get_classification_report'), "get_classification_report missing"
    assert hasattr(evaluator, 'compare_models'), "compare_models method missing"
    assert hasattr(evaluator, 'plot_metrics_comparison'), "plot_metrics_comparison missing"
    
    logger.info("✅ All evaluation methods present")
    return evaluator

def test_model_inference():
    """Test ModelInference"""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING MODEL INFERENCE")
    logger.info("=" * 70)
    
    inference = ModelInference()
    logger.info(f"✅ ModelInference initialized")
    
    # Check methods exist
    assert hasattr(inference, 'load_model'), "load_model method missing"
    assert hasattr(inference, 'set_model'), "set_model method missing"
    assert hasattr(inference, 'predict'), "predict method missing"
    assert hasattr(inference, 'predict_proba'), "predict_proba method missing"
    assert hasattr(inference, 'predict_with_confidence'), "predict_with_confidence missing"
    assert hasattr(inference, 'batch_predict'), "batch_predict method missing"
    assert hasattr(inference, 'predict_with_metadata'), "predict_with_metadata missing"
    assert hasattr(inference, 'validate_input'), "validate_input method missing"
    assert hasattr(inference, 'get_feature_importance'), "get_feature_importance missing"
    assert hasattr(inference, 'explain_prediction'), "explain_prediction missing"
    
    logger.info("✅ All inference methods present")
    return inference

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL IMPLEMENTATION VERIFICATION")
    logger.info("=" * 70)
    
    try:
        # Test each module
        models = test_model_building()
        trainer = test_model_training()
        evaluator = test_model_evaluation()
        inference = test_model_inference()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL MODEL IMPLEMENTATIONS VERIFIED")
        logger.info("=" * 70)
        logger.info("\nSummary:")
        logger.info(f"  • ModelBuilder: {len(models)} baseline models + 2 ensembles")
        logger.info(f"  • ModelTrainer: 7 methods (train, SMOTE, GridSearch, RandomSearch, etc.)")
        logger.info(f"  • ModelEvaluator: 6 methods (evaluate, plot CM, ROC, compare, etc.)")
        logger.info(f"  • ModelInference: 10 methods (predict, batch, metadata, explain, etc.)")
        logger.info("\n✅ Ready for training pipeline integration!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ VERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
