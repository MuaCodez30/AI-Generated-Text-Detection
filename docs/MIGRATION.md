# Migration Guide

This guide helps you understand the reorganization of the project structure.

## Changes Made

### 1. Utility Scripts Moved
Old root-level scripts have been moved to `scripts/` directory:
- `combiner.py` → `scripts/combine_datasets.py` (improved version)
- `combiner_human_ai.py` → `scripts/merge_data.py` (generalized version)
- `check_training.py` → `scripts/check_training.py` (moved and improved)

### 2. New Directories
- `scripts/`: All utility scripts
- `examples/`: Example data files and templates
- `docs/`: Additional documentation
- `data/`: Added README.md for documentation

### 3. Documentation Added
- `LICENSE`: MIT License
- `.gitignore`: Comprehensive Python gitignore
- Multiple README files in subdirectories
- Architecture and contributing documentation

## Updating Your Workflow

### If you used `combiner.py`:
```bash
# Old
python combiner.py

# New
python scripts/combine_datasets.py data/processed/human_clean.json data/processed/ai_clean.json data/combined/combined_dataset.json
```

### If you used `check_training.py`:
```bash
# Old
python check_training.py

# New (moved to scripts/)
python scripts/check_training.py
```

### If you used `tester.py`:
```bash
# Old
python tester.py

# New (use the prediction script)
python models/predict.py
```

## Cleanup (Optional)

The following files in the root directory can be removed after verifying the new scripts work:
- `combiner.py` (replaced by `scripts/combine_datasets.py`)
- `combiner_human_ai.py` (replaced by `scripts/merge_data.py`)
- `check_training.py` (moved to `scripts/check_training.py`)
- `models/train_test.py` (redundant, removed)
- `preprocessing/preprocess_combined.py` (redundant, removed)
- `preprocessing/preprocess_ai.py` (consolidated into `preprocessing/preprocess.py`)
- `preprocessing/preprocess_human.py` (consolidated into `preprocessing/preprocess.py`)
- `tester.py` (functionality available in `models/predict.py`)

**Note**: Review these files before deleting to ensure you don't lose any custom logic.
