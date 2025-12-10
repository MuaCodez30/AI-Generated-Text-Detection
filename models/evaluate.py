"""
Comprehensive evaluation script for AI text detection models.
"""
import json
import pickle
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from scipy.sparse import hstack
import matplotlib.pyplot as plt


class ModelEvaluator:
    """Evaluator class for AI text detection models."""
    
    def __init__(self, model_dir: str = "models/saved_models"):
        self.model_dir = model_dir
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.label_encoder = None
        self.svm_model = None
        
    def load_models(self):
        """Load trained models and vectorizers."""
        print("Loading models...")
        
        with open(os.path.join(self.model_dir, "word_vectorizer.pkl"), "rb") as f:
            self.word_vectorizer = pickle.load(f)
        
        with open(os.path.join(self.model_dir, "char_vectorizer.pkl"), "rb") as f:
            self.char_vectorizer = pickle.load(f)
        
        with open(os.path.join(self.model_dir, "svm_model.pkl"), "rb") as f:
            self.svm_model = pickle.load(f)
        
        with open(os.path.join(self.model_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        
        print("Models loaded successfully!")
    
    def evaluate_on_dataset(self, dataset_path: str):
        """Evaluate model on a dataset."""
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} samples from {dataset_path}")
        
        # Prepare data
        texts = []
        labels = []
        
        for article in dataset:
            content = article.get("content", "").strip()
            if content:
                texts.append(content)
                labels.append(article.get("label", "").lower())
        
        # Encode labels
        y_true = self.label_encoder.transform(labels)
        
        # Create features
        X_word = self.word_vectorizer.transform(texts)
        X_char = self.char_vectorizer.transform(texts)
        X = hstack([X_word, X_char])
        
        # Predict
        y_pred = self.svm_model.predict(X)
        y_proba = self.svm_model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        try:
            auc = roc_auc_score(y_true, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except:
            print("ROC-AUC: Could not calculate")
        
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=['Human', 'AI'],
            labels=[0, 1]
        ))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print(f"\nTrue Negatives (Human correctly identified): {cm[0][0]}")
        print(f"False Positives (Human misclassified as AI): {cm[0][1]}")
        print(f"False Negatives (AI misclassified as Human): {cm[1][0]}")
        print(f"True Positives (AI correctly identified): {cm[1][1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def predict_text(self, text: str):
        """Predict if a single text is AI or human generated."""
        if not self.svm_model:
            self.load_models()
        
        # Create features
        X_word = self.word_vectorizer.transform([text])
        X_char = self.char_vectorizer.transform([text])
        X = hstack([X_word, X_char])
        
        # Predict
        prediction = self.svm_model.predict(X)[0]
        probability = self.svm_model.predict_proba(X)[0]
        
        label = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probability[prediction] * 100
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'human': probability[0] * 100,
                'ai': probability[1] * 100
            }
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_models()
    
    # Evaluate on test set
    test_dataset = "data/combined/combined_dataset_new.json"
    if os.path.exists(test_dataset):
        results = evaluator.evaluate_on_dataset(test_dataset)
    else:
        print(f"Test dataset not found at {test_dataset}")


