"""
Prediction script for AI text detection.
Can be used to classify individual texts or batch files.
"""
import json
import pickle
import os
import sys
from scipy.sparse import hstack


class AITextDetector:
    """AI Text Detector for inference."""
    
    def __init__(self, model_dir: str = "models/saved_models"):
        self.model_dir = model_dir
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.label_encoder = None
        self.svm_model = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models."""
        try:
            with open(os.path.join(self.model_dir, "word_vectorizer.pkl"), "rb") as f:
                self.word_vectorizer = pickle.load(f)
            
            with open(os.path.join(self.model_dir, "char_vectorizer.pkl"), "rb") as f:
                self.char_vectorizer = pickle.load(f)
            
            with open(os.path.join(self.model_dir, "svm_model.pkl"), "rb") as f:
                self.svm_model = pickle.load(f)
            
            with open(os.path.join(self.model_dir, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)
            
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: Model files not found in {self.model_dir}")
            print("Please train the model first using train_model.py")
            sys.exit(1)
    
    def predict(self, text: str):
        """
        Predict if text is AI or human generated.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary with prediction and confidence scores
        """
        if not text.strip():
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'error': 'Empty text provided'
            }
        
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
            'confidence': round(confidence, 2),
            'probabilities': {
                'human': round(probability[0] * 100, 2),
                'ai': round(probability[1] * 100, 2)
            }
        }
    
    def predict_batch(self, texts: list):
        """
        Predict labels for a batch of texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    """Command-line interface for predictions."""
    detector = AITextDetector()
    
    if len(sys.argv) > 1:
        # File input
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "predictions.json"
        
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item.get("content", item.get("text", "")) for item in data]
        else:
            texts = [data.get("content", data.get("text", ""))]
        
        results = detector.predict_batch(texts)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Predictions saved to {output_file}")
    else:
        # Interactive mode
        print("AI Text Detector - Interactive Mode")
        print("Enter text to classify (or 'quit' to exit):\n")
        
        while True:
            text = input("Text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if text:
                result = detector.predict(text)
                print(f"\nPrediction: {result['label'].upper()}")
                print(f"Confidence: {result['confidence']}%")
                print(f"Probabilities - Human: {result['probabilities']['human']}%, AI: {result['probabilities']['ai']}%")
                print()


if __name__ == "__main__":
    main()


