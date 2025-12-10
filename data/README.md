# Data Directory

This directory contains all data files for the project.

## Structure

### `raw/`
Contains original, unprocessed data files.
- Raw text data from various sources
- Files are typically large and excluded from git

### `processed/`
Contains cleaned and preprocessed data files.
- Text normalization applied
- Ready for feature extraction
- Files are typically large and excluded from git

### `combined/`
Contains combined and balanced datasets ready for training.
- Merged human and AI datasets
- Balanced class distribution
- Labeled with ground truth
- Files are typically large and excluded from git

## Data Format

All data files should be in JSON format:

```json
[
  {
    "content": "Text content here...",
    "label": "human"  // or "ai"
  }
]
```

## Preprocessing

To preprocess raw data:

```bash
# Simple preprocessing (for both AI and human data)
python preprocessing/preprocess.py data/raw/ai.json data/processed/ai_clean.json
python preprocessing/preprocess.py data/raw/human.json data/processed/human_clean.json

# Advanced preprocessing (with extra cleaning features)
python preprocessing/advanced_preprocessing.py data/raw/dataset.json data/processed/dataset_clean.json
```

## Combining Datasets

To combine processed datasets:

```bash
python scripts/combine_datasets.py data/processed/human_clean.json data/processed/ai_clean.json data/combined/combined_dataset.json
```
