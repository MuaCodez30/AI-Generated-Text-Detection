# Architecture Overview

## System Architecture

The AI Text Detection system is built with a modular architecture that separates concerns into distinct components:

```
┌─────────────────┐
│   Data Layer    │
│  (Raw/Processed)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│     Pipeline    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│  & Evaluation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │
│     API         │
└─────────────────┘
```

## Components

### 1. Data Layer
- **Raw Data**: Original unprocessed text files
- **Processed Data**: Cleaned and normalized text
- **Combined Data**: Balanced datasets ready for training

### 2. Preprocessing Pipeline
- Text cleaning and normalization
- Handling of special characters
- Text standardization

### 3. Feature Engineering
- **Word-level TF-IDF**: Unigrams, bigrams, trigrams (25,000 features)
- **Character-level TF-IDF**: 3-6 character n-grams (35,000 features)
- **Total Features**: 60,000 combined features

### 4. Model Training
- Support Vector Machine (SVM) with linear kernel
- Logistic Regression (alternative model)
- Hyperparameter optimization
- Cross-validation

### 5. Prediction System
- Model loading and inference
- Batch and interactive prediction modes
- Probability scores

## Model Details

### Feature Extraction
- Uses scikit-learn's `TfidfVectorizer`
- Separate vectorizers for word and character n-grams
- Features are combined using sparse matrix stacking

### Classifiers
- **Primary**: SVM with linear kernel (optimized for high accuracy)
- **Secondary**: Logistic Regression (faster inference)

### Performance
- Accuracy: 99%+
- Balanced precision and recall
- High F1-scores for both classes
