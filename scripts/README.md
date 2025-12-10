# Utility Scripts

This directory contains utility scripts for data processing and project management.

## Available Scripts

### `combine_datasets.py`
Combines human and AI datasets for training. Balances datasets and creates labeled combined dataset.

**Usage:**
```bash
python scripts/combine_datasets.py <human_path> <ai_path> <output_path> [--no-balance]
```

**Example:**
```bash
python scripts/combine_datasets.py data/processed/human_clean.json data/processed/ai_clean.json data/combined/combined_dataset.json
```

### `merge_data.py`
Merges multiple JSON data files into a single file.

**Usage:**
```bash
python scripts/merge_data.py <input1.json> <input2.json> ... <output.json>
```

**Example:**
```bash
python scripts/merge_data.py data/processed/ai_clean.json data/processed/ai_clean_extra.json data/processed/ai_merged.json
```

### `check_training.py`
Checks training status and performs quick model evaluation.

**Usage:**
```bash
python scripts/check_training.py
```

This script:
- Verifies all required model files exist
- Shows model file sizes
- Runs a quick evaluation on the test dataset
