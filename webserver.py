import pickle
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Load Models on Startup ---
    print("Loading models...")
    base_path = Path("models") 
    
    try:
        with open(base_path / "word_vectorizer.pkl", "rb") as f:
            models["word_vect"] = pickle.load(f)
        with open(base_path / "char_vectorizer.pkl", "rb") as f:
            models["char_vect"] = pickle.load(f)
        with open(base_path / "svm_model.pkl", "rb") as f:
            models["svm"] = pickle.load(f)
            
        # 1. Get all feature names
        word_names = models["word_vect"].get_feature_names_out()
        char_names = models["char_vect"].get_feature_names_out()
        models["feature_names"] = np.concatenate([word_names, char_names])
        
        # 2. Store the split index to distinguish words vs chars later
        models["split_index"] = len(word_names)

        # 3. Pre-process coefficients
        if hasattr(models["svm"], "coef_"):
            models["coef"] = models["svm"].coef_.toarray().flatten() if hasattr(models["svm"].coef_, "toarray") else models["svm"].coef_.flatten()
        else:
            models["coef"] = None
            print("WARNING: SVM model has no coef_ attribute. Explanations will not work.")
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")
        raise e
        
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---

class TextRequest(BaseModel):
    text: str

class FeatureContribution(BaseModel):
    feature: str
    impact: float # The calculated score (tfidf * coef)

class Explanations(BaseModel):
    ai_words: list[FeatureContribution]
    human_words: list[FeatureContribution]
    ai_chars: list[FeatureContribution]
    human_chars: list[FeatureContribution]

class PredictionResponse(BaseModel):
    label: str
    confidence_score: float
    explanations: Explanations

# --- Helper to sort and slice ---
def get_top_k(items, k=10):
    # Sort by absolute impact (strength of contribution)
    return sorted(items, key=lambda x: abs(x['impact']), reverse=True)[:k]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1. Vectorize
    word_feat = models["word_vect"].transform([request.text])
    char_feat = models["char_vect"].transform([request.text])
    combined_feat = hstack([word_feat, char_feat])

    # 2. Predict (Standard SVM logic)
    # Mapping: 0 -> AI, 1 -> Human (Verify this matches your training labels!)
    pred_idx = models["svm"].predict(combined_feat)[0]
    probs = models["svm"].predict_proba(combined_feat)[0]
    
    label = "human" if pred_idx == 1 else "ai"
    confidence = float(max(probs))

    # 3. Calculate Explanations
    # Buckets for our four categories
    buckets = {
        "ai_words": [],
        "human_words": [],
        "ai_chars": [],
        "human_chars": []
    }

    if models["coef"] is not None:
        # Get indices where the input text actually has features (non-zero)
        nonzero_indices = combined_feat.nonzero()[1]
        feature_values = combined_feat.data # The TF-IDF values
        
        split_idx = models["split_index"]

        for i, feat_idx in enumerate(nonzero_indices):
            val = feature_values[i]
            coef = models["coef"][feat_idx]
            
            # Impact = How much this specific feature pushed the score
            # Negative Impact -> Pushes towards AI
            # Positive Impact -> Pushes towards Human
            impact = val * coef
            
            # Skip negligible impacts
            if abs(impact) < 1e-5:
                continue

            feature_name = models["feature_names"][feat_idx]
            item = {"feature": feature_name, "impact": impact}

            # Categorize
            is_word = feat_idx < split_idx
            is_human_leaning = impact > 0

            if is_word and is_human_leaning:
                buckets["human_words"].append(item)
            elif is_word and not is_human_leaning:
                buckets["ai_words"].append(item)
            elif not is_word and is_human_leaning:
                buckets["human_chars"].append(item)
            else: # char and ai leaning
                buckets["ai_chars"].append(item)

    # 4. Construct Final Response with Top 10s
    return {
        "label": label,
        "confidence_score": confidence,
        "explanations": {
            "ai_words": get_top_k(buckets["ai_words"]),
            "human_words": get_top_k(buckets["human_words"]),
            "ai_chars": get_top_k(buckets["ai_chars"]),
            "human_chars": get_top_k(buckets["human_chars"])
        }
    }
