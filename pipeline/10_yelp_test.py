import requests
import json

# --- CONFIGURATION ---
# This is your actual key
API_KEY = "RVn7dRo9i2hByWH4_fL82iU8REmuQfP4mCqA9_xJNWZUSx-zYxL3bwSiYt2ssBGGPoih4iReUDWBS_L1tELz_0381-heq47j3JgUMNtEatzyg91vzLh4GHSpXapZaXYx"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
ENDPOINT = "https://api.yelp.com/v3/businesses/search"


def search_yelp(term, location="Milpitas, CA"):
    print(f"ORBIT SYSTEM: Scanning Yelp for '{term}' near '{location}'...")

    params = {"term": term, "location": location, "limit": 3, "sort_by": "best_match"}

    try:
        response = requests.get(ENDPOINT, headers=HEADERS, params=params)

        if response.status_code == 200:
            data = response.json()
            businesses = data.get("businesses", [])

            print(f"\nSTATUS: Success. Found {len(businesses)} results.")
            for b in businesses:
                print("-" * 40)
                print(f"NAME:    {b['name']}")
                print(f"RATING:  {b['rating']} Stars ({b['review_count']} reviews)")
                print(f"ADDRESS: {', '.join(b['location']['display_address'])}")
                print(
                    f"COORDS:  {b['coordinates']['latitude']}, {b['coordinates']['longitude']}"
                )

                categories = [c["title"] for c in b["categories"]]
                print(f"TAGS:    {', '.join(categories)}")
        else:
            print(f"ERROR: Yelp rejected the connection. Code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")


# --- TEST RUN ---
# We are specifically searching for what failed last time
print("\n--- TEST 1: CLOTHING AT GREAT MALL ---")
search_yelp("Uniqlo", "Great Mall, Milpitas")

print("\n--- TEST 2: FOOD NEAR LIBRARY ---")
search_yelp("Burger", "Milpitas Library")
