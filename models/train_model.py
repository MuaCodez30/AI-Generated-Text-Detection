import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset (970 AI + 970 human)
with open("data/processed/combined_dataset_clean.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Optionally downsample human articles to match AI
ai_articles = [a for a in dataset if a["label"] == "AI"]
human_articles = [a for a in dataset if a["label"] == "human"][:len(ai_articles)]
balanced_dataset = ai_articles + human_articles

texts = [article["content"] for article in balanced_dataset]
labels = [article["label"] for article in balanced_dataset]  # "AI" or "human"

# Encode labels as integers
le = LabelEncoder()
y = le.fit_transform(labels)  # 0 = human, 1 = AI

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # top 5000 words
X = vectorizer.fit_transform(texts)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")