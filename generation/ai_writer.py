# ai_generation/generate_ai_articles_resume.py
import json
import os
import time
import openai

# Paths
INPUT_PATH = "data/processed/human_clean.json"
OUTPUT_PATH = "data/raw/ai.json"

# OpenAI settings
openai.api_key = os.getenv("OPENAI_API_KEY")  # set your API key in environment variable
MODEL = "gpt-3.5-turbo"                       # or "gpt-4" if you have access
DELAY = 3  # seconds delay between requests to avoid rate limits

# Load human dataset
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    human_articles = json.load(f)

# Load previously generated AI articles if any
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        ai_articles = json.load(f)
        processed_titles = {a["title"] for a in ai_articles}
else:
    ai_articles = []
    processed_titles = set()

print(f"Loaded {len(human_articles)} human articles")
print(f"{len(processed_titles)} articles already processed")

def generate_ai_content(title):
    prompt = f"Write a news article in Azerbaijani based on this title:\n\n{title}\n\nFull article:"
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating AI content for '{title}': {e}")
        return None

# Main loop
for idx, article in enumerate(human_articles, start=1):
    title = article["title"]
    
    if title in processed_titles:
        print(f"[{idx}/{len(human_articles)}] Already processed, skipping: {title[:50]}...")
        continue

    print(f"[{idx}/{len(human_articles)}] Generating AI article for: {title[:50]}...")
    
    content = None
    retries = 3
    while retries > 0 and content is None:
        content = generate_ai_content(title)
        if content is None:
            retries -= 1
            print(f"  Retry remaining: {retries}")
            time.sleep(DELAY)

    if content:
        ai_articles.append({
            "title": title,
            "content": content
        })
        processed_titles.add(title)

        # Save incrementally after each successful generation
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(ai_articles, f, ensure_ascii=False, indent=2)
        print(f"  Saved successfully!")
    else:
        print(f"  Skipped '{title}' after 3 failed attempts")

    time.sleep(DELAY)

print(f"Finished! Total AI-generated articles saved: {len(ai_articles)}")

