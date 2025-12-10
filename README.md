# AI Text Generation Detector

A sophisticated machine learning system for detecting AI-generated text using advanced natural language processing techniques.

## 🚀 Features

- **Advanced Feature Engineering**: Combines word-level and character-level n-gram features
- **High Accuracy**: SVM model optimized to achieve 99%+ accuracy
- **Comprehensive Preprocessing**: Advanced text cleaning and normalization
- **Production Ready**: Includes training, evaluation, and prediction scripts
- **Modular Architecture**: Well-organized codebase for easy maintenance and extension

## 📁 Project Structure

```
SDP_DRAFT/
├── data/                      # Data directory
│   ├── raw/                   # Raw input data files
│   ├── processed/             # Preprocessed/cleaned data
│   └── combined/              # Combined datasets for training
│
├── preprocessing/             # Data preprocessing scripts
│   ├── preprocess.py          # Unified preprocessing script
│   └── advanced_preprocessing.py  # Advanced preprocessing with extra features
│
├── models/                    # Machine learning models
│   ├── train_model.py         # Main training script
│   ├── evaluate.py            # Model evaluation
│   ├── predict.py             # Prediction/inference
│   └── saved_models/          # Trained model files
│       └── README.md          # Model documentation
│
├── scripts/                   # Utility scripts
│   ├── combine_datasets.py    # Combine human/AI datasets
│   ├── merge_data.py          # Merge multiple data files
│   ├── check_training.py      # Check training status
│   └── README.md              # Scripts documentation
│
├── scraping/                  # Web scraping utilities
│   ├── scraper.py
│   └── utils.py
│
├── generation/                # AI content generation
│   └── ai_writer.py
│
├── utils/                     # General utilities
│   ├── __init__.py
│   └── text_utils.py
│
├── notebooks/                 # Jupyter notebooks
│   └── exploration.ipynb      # Data exploration
│
├── examples/                  # Example files
│   ├── example_data.json      # Sample data format
│   └── README.md
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   └── CONTRIBUTING.md        # Contribution guidelines
│
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SDP_DRAFT.git
   cd SDP_DRAFT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Training the Model

Train the AI text detection model:

```bash
python models/train_model.py
```

The script will:
- Load and balance the dataset
- Create word and character n-gram features
- Train SVM and Logistic Regression models
- Save models to `models/saved_models/`

### Evaluating the Model

Evaluate the trained model:

```bash
python models/evaluate.py
```

### Making Predictions

#### Interactive Mode
```bash
python models/predict.py
```

#### Batch Prediction
```bash
python models/predict.py input.json output.json
```

### Preprocessing Data

Preprocess raw data files:

**Simple preprocessing:**
```bash
python preprocessing/preprocess.py data/raw/ai.json data/processed/ai_clean.json
python preprocessing/preprocess.py data/raw/human.json data/processed/human_clean.json
```

**Advanced preprocessing (with extra cleaning):**
```bash
python preprocessing/advanced_preprocessing.py data/raw/dataset.json data/processed/dataset_clean.json
```

### Utility Scripts

Combine datasets:
```bash
python scripts/combine_datasets.py data/processed/human_clean.json data/processed/ai_clean.json data/combined/combined_dataset.json
```

Check training status:
```bash
python scripts/check_training.py
```

## 🏗️ Model Architecture

The model uses:
- **Word-level TF-IDF**: Unigrams, bigrams, and trigrams (25,000 features)
- **Character-level TF-IDF**: 3-6 character n-grams (35,000 features)
- **SVM Classifier**: Linear kernel with optimized hyperparameters
- **Total Features**: 60,000 combined features

For more details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 📊 Performance

The model achieves:
- **Accuracy**: 99%+
- **Precision**: High precision for both classes
- **Recall**: High recall for both classes
- **F1-Score**: Balanced performance

## 📝 Data Format

Input data should be in JSON format:

```json
[
  {
    "content": "Text content here...",
    "label": "human"  // or "ai"
  }
]
```

See [examples/example_data.json](examples/example_data.json) for a sample file.

## 🤝 Contributing

Contributions are welcome! Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Additional Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Data Directory](data/README.md)
- [Saved Models](models/saved_models/README.md)
- [Utility Scripts](scripts/README.md)

## 🔍 Project Status

This project is actively maintained and ready for use. For issues or questions, please open an issue on GitHub.
