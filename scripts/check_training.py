"""
Quick script to check training status and evaluate model.
"""
import os
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evaluate import ModelEvaluator


def check_training_status(model_dir: str = "models/saved_models"):
    """Check if training has completed."""
    required_files = [
        "word_vectorizer.pkl",
        "char_vectorizer.pkl",
        "svm_model.pkl",
        "label_encoder.pkl"
    ]
    
    all_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    
    if all_exist:
        print("✓ All model files exist!")
        
        # Check file sizes
        print("\nModel file sizes:")
        for f in required_files:
            path = os.path.join(model_dir, f)
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"  {f}: {size:.2f} MB")
        
        return True
    else:
        print("Training not complete. Missing files:")
        for f in required_files:
            path = os.path.join(model_dir, f)
            if not os.path.exists(path):
                print(f"  ✗ {f}")
        return False


def quick_evaluate(test_dataset: str = "data/combined/combined_dataset_new.json"):
    """Quick evaluation of the model."""
    print("\n" + "=" * 60)
    print("Quick Model Evaluation")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    evaluator.load_models()
    
    # Evaluate on test set
    if os.path.exists(test_dataset):
        results = evaluator.evaluate_on_dataset(test_dataset)
        return results
    else:
        print(f"Test dataset not found at {test_dataset}")
        return None


if __name__ == "__main__":
    if check_training_status():
        print("\nRunning evaluation...")
        results = quick_evaluate()
        if results:
            print(f"\n✓ Final Accuracy: {results['accuracy']*100:.2f}%")
    else:
        print("\nWaiting for training to complete...")
