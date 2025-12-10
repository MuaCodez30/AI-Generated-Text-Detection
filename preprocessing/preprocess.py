"""
Unified preprocessing script for AI and human text data.
Cleans and normalizes text from raw JSON files.
"""
import json
import re
import os
import sys
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean article text: remove extra whitespace, HTML entities, and special characters."""
    if not isinstance(text, str):
        return ""
    
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Replace non-breaking spaces
    text = text.replace('\xa0', ' ')
    
    return text.strip()


def preprocess_article(article: dict) -> dict:
    """Preprocess a single article dictionary."""
    return {
        "title": clean_text(article.get("title", "")),
        "content": clean_text(article.get("content", "")),
    }


def preprocess_dataset(input_path: str, output_path: str) -> int:
    """
    Preprocess a dataset from input file to output file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    
    Returns:
        Number of processed articles
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw dataset
    print(f"Loading articles from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)
    
    print(f"Loaded {len(raw_articles)} articles")
    
    # Preprocess each article
    clean_articles = [preprocess_article(a) for a in raw_articles]
    
    # Save cleaned dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(clean_articles)} cleaned articles to {output_path}")
    return len(clean_articles)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocessing/preprocess.py <input_path> <output_path>")
        print("\nExample:")
        print("  python preprocessing/preprocess.py data/raw/ai.json data/processed/ai_clean.json")
        print("  python preprocessing/preprocess.py data/raw/human.json data/processed/human_clean.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    preprocess_dataset(input_path, output_path)
