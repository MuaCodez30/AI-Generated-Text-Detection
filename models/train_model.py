"""
Advanced AI Text Detection Model Training
Uses combined word and character n-gram features with optimized SVM.
"""
import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.utils import shuffle
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')


class AITextDetectorTrainer:
    """Trainer class for AI text detection models."""
    
    def __init__(self, dataset_path: str, model_save_dir: str = "models/saved_models"):
        self.dataset_path = dataset_path
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.label_encoder = None
        self.svm_model = None
        self.lr_model = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("=" * 60)
        print("Loading Dataset")
        print("=" * 60)
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        print(f"Total samples loaded: {len(dataset)}")
        
        # Normalize labels
        for article in dataset:
            article["label"] = article["label"].lower()
        
        # Balance dataset
        ai_articles = [a for a in dataset if a["label"] == "ai"]
        human_articles = [a for a in dataset if a["label"] == "human"]
        
        print(f"Before balancing - AI: {len(ai_articles)}, Human: {len(human_articles)}")
        
        n = min(len(ai_articles), len(human_articles))
        ai_articles = ai_articles[:n]
        human_articles = human_articles[:n]
        balanced_dataset = ai_articles + human_articles
        balanced_dataset = shuffle(balanced_dataset, random_state=42)
        
        print(f"After balancing - AI: {len(ai_articles)}, Human: {len(human_articles)}")
        print(f"Total balanced: {len(balanced_dataset)}")
        
        # Extract texts & labels
        texts = []
        labels = []
        
        for article in balanced_dataset:
            content = article.get("content", "").strip()
            if content:  # Only include non-empty texts
                texts.append(content)
                labels.append(article["label"])
        
        print(f"Final dataset size after filtering: {len(texts)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        print(f"Label classes: {self.label_encoder.classes_}")
        print(f"Label distribution: AI={np.sum(y==1)}, Human={np.sum(y==0)}")
        
        return texts, y
    
    def create_features(self, texts):
        """Create combined word and character n-gram features."""
        print("\n" + "=" * 60)
        print("Creating Features")
        print("=" * 60)
        
        # Word-level TF-IDF with n-grams
        print("Creating word-level features...")
        self.word_vectorizer = TfidfVectorizer(
            max_features=20000,      # Optimized for speed and accuracy
            ngram_range=(1, 3),      # Unigrams, bigrams, trigrams
            sublinear_tf=True,       # Apply sublinear TF scaling
            min_df=2,                # Minimum document frequency
            max_df=0.95,             # Maximum document frequency
            analyzer='word',
            lowercase=True
        )
        
        X_word = self.word_vectorizer.fit_transform(texts)
        print(f"Word features shape: {X_word.shape}")
        
        # Character-level TF-IDF (critical for AI detection)
        print("Creating character-level features...")
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),      # 3-5 character n-grams (optimized)
            max_features=30000,      # Optimized char features
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )
        
        X_char = self.char_vectorizer.fit_transform(texts)
        print(f"Character features shape: {X_char.shape}")
        
        # Combine both feature sets
        X = hstack([X_word, X_char])
        print(f"Combined feature matrix shape: {X.shape}")
        print(f"Total features: {X.shape[1]:,}")
        
        return X
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train both Logistic Regression and SVM models."""
        print("\n" + "=" * 60)
        print("Training Models")
        print("=" * 60)
        
        # Train Logistic Regression baseline (optional, can skip for speed)
        print("\n--- Training Logistic Regression (Baseline) ---")
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='lbfgs',
            n_jobs=-1
        )
        self.lr_model.fit(X_train, y_train)
        
        y_pred_lr = self.lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, y_pred_lr)
        
        print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
        
        # Train SVM with hyperparameter tuning
        print("\n--- Training SVM with Hyperparameter Tuning ---")
        
        # Optimized hyperparameter tuning for faster training
        print("Tuning linear SVM...")
        
        # Use a smaller sample for grid search to speed up
        if X_train.shape[0] > 8000:
            sample_size = 8000
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
            print(f"Using {sample_size} samples for grid search (for speed)...")
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        # Focused grid search for linear kernel
        linear_params = {'C': [0.5, 1.0, 2.0, 5.0, 10.0]}
        linear_svm = SVC(kernel='linear', probability=True, random_state=42)
        
        linear_grid = GridSearchCV(
            linear_svm,
            linear_params,
            cv=3,  # Reduced CV folds for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running grid search...")
        linear_grid.fit(X_train_sample, y_train_sample)
        print(f"Best linear SVM parameters: {linear_grid.best_params_}")
        print(f"Best CV score: {linear_grid.best_score_:.4f}")
        
        # Train final model on full training set with best parameters
        best_C = linear_grid.best_params_['C']
        print(f"\nTraining final SVM on full training set with C={best_C}...")
        self.svm_model = SVC(
            kernel='linear',
            C=best_C,
            probability=True,
            random_state=42
        )
        
        self.svm_model.fit(X_train, y_train)
        
        # Evaluate SVM
        y_pred_svm = self.svm_model.predict(X_test)
        y_proba_svm = self.svm_model.predict_proba(X_test)[:, 1]
        
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        
        print(f"\n--- Final SVM Results ---")
        print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_svm, target_names=['Human', 'AI']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_svm)
        print(cm)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        try:
            auc = roc_auc_score(y_test, y_proba_svm)
            print(f"ROC-AUC: {auc:.4f}")
        except:
            print("ROC-AUC: Could not calculate (binary classification issue)")
        
        return svm_accuracy
    
    def save_models(self):
        """Save all trained models and vectorizers."""
        print("\n" + "=" * 60)
        print("Saving Models")
        print("=" * 60)
        
        # Save vectorizers
        with open(os.path.join(self.model_save_dir, "word_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.word_vectorizer, f)
        print("Saved word_vectorizer.pkl")
        
        with open(os.path.join(self.model_save_dir, "char_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.char_vectorizer, f)
        print("Saved char_vectorizer.pkl")
        
        # Save models
        with open(os.path.join(self.model_save_dir, "svm_model.pkl"), "wb") as f:
            pickle.dump(self.svm_model, f)
        print("Saved svm_model.pkl")
        
        with open(os.path.join(self.model_save_dir, "logreg_model.pkl"), "wb") as f:
            pickle.dump(self.lr_model, f)
        print("Saved logreg_model.pkl")
        
        # Save label encoder
        with open(os.path.join(self.model_save_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        print("Saved label_encoder.pkl")
        
        print(f"\nAll models saved to: {self.model_save_dir}")
    
    def run(self):
        """Run the complete training pipeline."""
        # Load and prepare data
        texts, y = self.load_and_prepare_data()
        
        # Create features
        X = self.create_features(texts)
        
        # Split data
        print("\n" + "=" * 60)
        print("Splitting Data")
        print("=" * 60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Train labels - AI: {np.sum(y_train==1)}, Human: {np.sum(y_train==0)}")
        print(f"Test labels - AI: {np.sum(y_test==1)}, Human: {np.sum(y_test==0)}")
        
        # Train models
        accuracy = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final SVM Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy


if __name__ == "__main__":
    # Default dataset path
    dataset_path = "data/combined/combined_dataset_new.json"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset exists or update the path.")
        exit(1)
    
    # Create trainer and run
    trainer = AITextDetectorTrainer(dataset_path)
    accuracy = trainer.run()
