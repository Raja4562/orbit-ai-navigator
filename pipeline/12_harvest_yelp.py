import requests
import json
import time

# --- CONFIGURATION ---
API_KEY = "RVn7dRo9i2hByWH4_fL82iU8REmuQfP4mCqA9_xJNWZUSx-zYxL3bwSiYt2ssBGGPoih4iReUDWBS_L1tELz_0381-heq47j3JgUMNtEatzyg91vzLh4GHSpXapZaXYx"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
ENDPOINT = "https://api.yelp.com/v3/businesses/search"
OUTPUT_FILE = "milpitas_rich.json"

# Expanded Categories for Maximum Data
CATEGORIES = ["restaurants", "shopping", "active", "arts", "nightlife", "beauty"]


def harvest_category(category):
    print(f"\n--- HARVESTING: {category.upper()} ---")
    all_businesses = []
    offset = 0

    # Yelp's hard limit is 1000 results per search query.
    while offset < 1000:
        params = {
            "location": "Milpitas, CA",
            "categories": category,
            "limit": 50,  # Max per call
            "offset": offset,
            "sort_by": "review_count",
        }

        try:
            response = requests.get(ENDPOINT, headers=HEADERS, params=params)

            # If we hit a rate limit or error, stop this category
            if response.status_code != 200:
                print(f"   Stopping: Server returned {response.status_code}")
                break

            data = response.json()
            batch = data.get("businesses", [])

            # If no more results came back, we are done
            if not batch:
                print("   -> Reached end of list.")
                break

            for b in batch:
                # Feature Engineering: Extract Clean Data
                item = {
                    "id": b["id"],
                    "name": b["name"],
                    "rating": b["rating"],
                    "review_count": b["review_count"],
                    "price": b.get("price", "?"),
                    "address": ", ".join(b["location"]["display_address"]),
                    "coords": [
                        b["coordinates"]["latitude"],
                        b["coordinates"]["longitude"],
                    ],
                    "categories": [c["title"] for c in b["categories"]],
                    "url": b["url"],
                }
                all_businesses.append(item)

            print(f"   Collected {len(batch)} items (Total: {offset + len(batch)})...")
            offset += 50

            # Sleep 1.2s to prevent Yelp from blocking your IP
            time.sleep(1.2)

        except Exception as e:
            print(f"   Error: {e}")
            break

    return all_businesses


# --- MAIN EXECUTION ---
master_db = []
seen_ids = set()

print("ORBIT SYSTEM: Starting MAX Data Harvest...")
print("NOTE: This will take a few minutes. Do not close.")

for cat in CATEGORIES:
    results = harvest_category(cat)

    new_count = 0
    for item in results:
        # Deduplication: Many restaurants are also listed under 'nightlife', etc.
        if item["id"] not in seen_ids:
            master_db.append(item)
            seen_ids.add(item["id"])
            new_count += 1

    print(f"   -> Added {new_count} unique locations to database.")

# --- SAVE ---
with open(OUTPUT_FILE, "w") as f:
    json.dump(master_db, f, indent=4)

print(f"\nSUCCESS: Max Harvest Complete.")
print(f"Total Unique Locations in Database: {len(master_db)}")
print(f"Saved to '{OUTPUT_FILE}'")
