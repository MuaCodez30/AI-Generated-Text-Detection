import json
import re

# Load combined dataset
with open("data/raw/combined_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters and numbers (keep Azerbaijani letters)
    text = re.sub(r"[^a-zA-ZəƏçÇşŞıİğĞöÖüÜ\s]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

for article in dataset:
    article["content"] = clean_text(article["content"])

# Save cleaned dataset
with open("data/processed/combined_dataset_clean.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Cleaned dataset saved! Total articles: {len(dataset)}")