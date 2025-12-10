"""
Advanced preprocessing module for AI text detection.
Includes comprehensive text cleaning and normalization.
"""
import json
import re
import os
import sys
from typing import Dict, Any


def clean_text(text: str) -> str:
    """
    Advanced text cleaning for AI detection.
    Preserves linguistic patterns while removing noise.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize whitespace (preserve single spaces)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-breaking spaces and special unicode
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    
    # Remove excessive punctuation (keep single punctuation)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()


def preprocess_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single article.
    
    Args:
        article: Dictionary with 'content' and optionally 'title' and 'label'
    
    Returns:
        Preprocessed article dictionary
    """
    content = article.get("content", "")
    title = article.get("title", "")
    
    # Combine title and content if both exist
    full_text = f"{title} {content}".strip() if title else content
    
    # Clean the text
    cleaned_text = clean_text(full_text)
    
    # Build preprocessed article
    preprocessed = {
        "content": cleaned_text,
    }
    
    # Add label if present
    if "label" in article:
        preprocessed["label"] = article["label"].lower()
    
    return preprocessed


def preprocess_dataset(input_path: str, output_path: str) -> int:
    """
    Preprocess an entire dataset.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    
    Returns:
        Number of processed articles
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw dataset
    with open(input_path, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)
    
    print(f"Loaded {len(raw_articles)} articles from {input_path}")
    
    # Preprocess each article
    preprocessed_articles = []
    for article in raw_articles:
        preprocessed = preprocess_article(article)
        # Only keep articles with non-empty content
        if preprocessed["content"].strip():
            preprocessed_articles.append(preprocessed)
    
    # Save preprocessed dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(preprocessed_articles)} preprocessed articles to {output_path}")
    return len(preprocessed_articles)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocessing/advanced_preprocessing.py <input_path> <output_path>")
        print("\nExample:")
        print("  python preprocessing/advanced_preprocessing.py data/raw/dataset.json data/processed/dataset_clean.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    preprocess_dataset(input_path, output_path)


