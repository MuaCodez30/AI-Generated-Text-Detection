# Saved Models

This directory contains trained model files.

## Model Files

The following model files are saved here after training:

- `word_vectorizer.pkl`: Word-level TF-IDF vectorizer
- `char_vectorizer.pkl`: Character-level TF-IDF vectorizer
- `svm_model.pkl`: Trained SVM classifier
- `logreg_model.pkl`: Trained Logistic Regression classifier
- `label_encoder.pkl`: Label encoder for class labels

## Loading Models

All models are saved as pickle files (.pkl) and can be loaded using:

```python
import pickle
import os

model_dir = "models/saved_models"

with open(os.path.join(model_dir, "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)
```

## Training Models

Model files are large and are excluded from git by default (see `.gitignore`). 
To use the models, you need to train them first using:

```bash
python models/train_model.py
```

This will create all required model files in this directory.
