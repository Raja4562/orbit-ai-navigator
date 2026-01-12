import requests
import json
import time

# --- CONFIGURATION ---
API_KEY = "RVn7dRo9i2hByWH4_fL82iU8REmuQfP4mCqA9_xJNWZUSx-zYxL3bwSiYt2ssBGGPoih4iReUDWBS_L1tELz_0381-heq47j3JgUMNtEatzyg91vzLh4GHSpXapZaXYx"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
INPUT_FILE = "milpitas_rich.json"
OUTPUT_FILE = "milpitas_ultra.json"

MAX_REVIEWS_TO_FETCH = 50


def get_text_reviews(biz_id):
    url = f"https://api.yelp.com/v3/businesses/{biz_id}/reviews"
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            return [item["text"] for item in data.get("reviews", [])]
        else:
            print(f"   [API ERROR {r.status_code}]: {r.text[:50]}...")
    except Exception as e:
        print(f"   [CRITICAL]: {e}")
    return []


# --- MAIN ---
print("ORBIT SYSTEM: Injecting Text Reviews (ID Mode)...")

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# Sort: Most popular places get reviews first
data.sort(key=lambda x: x["review_count"], reverse=True)

enhanced_data = []
count = 0

for place in data:
    if count < MAX_REVIEWS_TO_FETCH:
        print(f"[{count+1}/{MAX_REVIEWS_TO_FETCH}] Reviews for: {place['name']}...")

        # Now using the OFFICIAL ID
        reviews = get_text_reviews(place["id"])
        place["text_reviews"] = reviews

        if reviews:
            print(f"   -> Success: Got {len(reviews)} reviews.")

        count += 1
        time.sleep(1.0)
    else:
        place["text_reviews"] = []

    enhanced_data.append(place)

with open(OUTPUT_FILE, "w") as f:
    json.dump(enhanced_data, f, indent=4)

print(f"\nSUCCESS: Ultra Data saved to '{OUTPUT_FILE}'.")
