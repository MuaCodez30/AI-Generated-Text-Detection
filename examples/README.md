# Examples

This directory contains example files to help you get started with the project.

## Files

- `example_data.json`: Sample data file showing the expected format for training data

## Data Format

The project expects JSON files in the following format:

```json
[
  {
    "content": "Your text content here...",
    "label": "human"  // or "ai"
  }
]
```

Each entry should have:
- `content`: The text to be classified (string)
- `label`: The ground truth label - either "human" or "ai" (string)

## Usage

You can use these examples as templates when preparing your own datasets.
