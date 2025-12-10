"""
Combine human and AI datasets for training.
Balances the datasets and creates a labeled combined dataset.
"""
import json
import sys
from pathlib import Path
from random import shuffle, sample


def combine_datasets(human_path: str, ai_path: str, output_path: str, balance: bool = True):
    """
    Combine human and AI datasets.
    
    Args:
        human_path: Path to human text JSON file
        ai_path: Path to AI text JSON file
        output_path: Path to save combined dataset
        balance: If True, balance datasets to match smaller size
    """
    # Load datasets
    print(f"Loading human articles from {human_path}...")
    with open(human_path, "r", encoding="utf-8") as f:
        human_articles = json.load(f)
    
    print(f"Loading AI articles from {ai_path}...")
    with open(ai_path, "r", encoding="utf-8") as f:
        ai_articles = json.load(f)
    
    print(f"Human articles: {len(human_articles)}")
    print(f"AI articles: {len(ai_articles)}")
    
    # Balance if requested
    if balance:
        min_count = min(len(human_articles), len(ai_articles))
        if len(human_articles) > min_count:
            print(f"Downsampling human articles to {min_count}...")
            human_articles = sample(human_articles, min_count)
        if len(ai_articles) > min_count:
            print(f"Downsampling AI articles to {min_count}...")
            ai_articles = sample(ai_articles, min_count)
    
    # Label articles
    human_labeled = [
        {"content": a.get("content", a.get("text", "")), "label": "human"} 
        for a in human_articles
    ]
    ai_labeled = [
        {"content": a.get("content", a.get("text", "")), "label": "ai"} 
        for a in ai_articles
    ]
    
    # Combine and shuffle
    dataset = human_labeled + ai_labeled
    shuffle(dataset)
    
    # Save combined dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Combined dataset saved to {output_path}")
    print(f"  Total articles: {len(dataset)}")
    print(f"  Human: {len(human_labeled)}, AI: {len(ai_labeled)}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/combine_datasets.py <human_path> <ai_path> <output_path> [--no-balance]")
        print("\nExample:")
        print("  python scripts/combine_datasets.py data/processed/human_clean.json data/processed/ai_clean.json data/combined/combined_dataset.json")
        sys.exit(1)
    
    human_path = sys.argv[1]
    ai_path = sys.argv[2]
    output_path = sys.argv[3]
    balance = "--no-balance" not in sys.argv
    
    combine_datasets(human_path, ai_path, output_path, balance)
