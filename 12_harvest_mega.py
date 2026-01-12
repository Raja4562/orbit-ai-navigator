import requests
import json
import time

# --- CONFIGURATION ---
API_KEY = "RVn7dRo9i2hByWH4_fL82iU8REmuQfP4mCqA9_xJNWZUSx-zYxL3bwSiYt2ssBGGPoih4iReUDWBS_L1tELz_0381-heq47j3JgUMNtEatzyg91vzLh4GHSpXapZaXYx"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
ENDPOINT = "https://api.yelp.com/v3/businesses/search"
OUTPUT_FILE = "milpitas_rich.json"

# LOCATIONS TO SCAN (Milpitas + immediate borders to hit 5k)
TARGET_ZONES = ["Milpitas, CA", "North San Jose, CA"]

# THE MEGA LIST (100+ Sectors)
SEARCH_TERMS = [
    # --- FOOD & DRINK ---
    "chinese",
    "mexican",
    "italian",
    "japanese",
    "korean",
    "vietnamese",
    "indian",
    "thai",
    "mediterranean",
    "halal",
    "vegetarian",
    "vegan",
    "burgers",
    "pizza",
    "seafood",
    "steak",
    "sushi",
    "dim sum",
    "ramen",
    "tacos",
    "bbq",
    "sandwiches",
    "coffee",
    "tea",
    "bubble tea",
    "juice bars",
    "breweries",
    "wineries",
    "bars",
    "pubs",
    "cocktail bars",
    "sports bars",
    "ice cream",
    "desserts",
    "bakeries",
    "donuts",
    "food trucks",
    "grocery",
    "convenience stores",
    "farmers market",
    # --- SHOPPING ---
    "fashion",
    "men's clothing",
    "women's clothing",
    "shoes",
    "jewelry",
    "watches",
    "electronics",
    "computers",
    "mobile phones",
    "cameras",
    "video games",
    "furniture",
    "home decor",
    "mattresses",
    "appliances",
    "hardware",
    "tools",
    "department stores",
    "outlets",
    "shopping centers",
    "drugstores",
    "cosmetics",
    "beauty supply",
    "bookstores",
    "toys",
    "hobbies",
    "florists",
    "gift shops",
    "sporting goods",
    "bikes",
    "musical instruments",
    "office equipment",
    # --- ACTIVE & FUN ---
    "gyms",
    "yoga",
    "pilates",
    "crossfit",
    "martial arts",
    "boxing",
    "dance studios",
    "personal trainers",
    "parks",
    "hiking",
    "playgrounds",
    "golf",
    "tennis",
    "swimming pools",
    "bowling",
    "movie theaters",
    "arcades",
    "escape games",
    "mini golf",
    "museums",
    "landmarks",
    "festivals",
    # --- HEALTH & MEDICAL ---
    "doctors",
    "dentists",
    "orthodontists",
    "optometrists",
    "chiropractors",
    "physical therapy",
    "massage",
    "acupuncture",
    "urgent care",
    "hospitals",
    "pharmacies",
    "nutritionists",
    "psychologists",
    "counseling",
    # --- BEAUTY & SPAS ---
    "hair salons",
    "barbers",
    "nail salons",
    "hair removal",
    "skin care",
    "tanning",
    "spas",
    "tattoo",
    "piercing",
    "makeup artists",
    # --- AUTOMOTIVE ---
    "auto repair",
    "oil change",
    "tires",
    "car wash",
    "auto detailing",
    "body shops",
    "car dealers",
    "motorcycle repair",
    "towing",
    "gas stations",
    "ev charging",
    "parking",
    # --- HOME SERVICES ---
    "plumbers",
    "electricians",
    "hvac",
    "contractors",
    "painters",
    "roofing",
    "flooring",
    "carpet cleaning",
    "house cleaning",
    "landscaping",
    "gardeners",
    "pest control",
    "movers",
    "storage",
    "locksmiths",
    "garage door",
    "internet providers",
    "security systems",
    # --- PROFESSIONAL SERVICES ---
    "lawyers",
    "accountants",
    "tax services",
    "real estate",
    "mortgage brokers",
    "insurance",
    "financial advising",
    "banks",
    "notaries",
    "printing",
    "shipping",
    "it services",
    "web design",
    "marketing",
    "consulting",
    # --- EDUCATION & KIDS ---
    "preschools",
    "child care",
    "tutoring",
    "test prep",
    "music lessons",
    "art classes",
    "cooking classes",
    "driving schools",
    # --- PETS ---
    "veterinarians",
    "pet grooming",
    "pet boarding",
    "pet sitting",
    "dog walkers",
    "pet stores",
    "animal shelters",
]


def harvest_sector(term, location):
    print(f"   Scanning '{term.upper()}' in {location}...")
    term_businesses = []
    offset = 0

    # Loop up to 150 per sector (Usually saturates the specific niche)
    while offset < 150:
        params = {
            "location": location,
            "term": term,
            "limit": 50,
            "offset": offset,
            "sort_by": "review_count",
        }

        try:
            response = requests.get(ENDPOINT, headers=HEADERS, params=params)
            if response.status_code != 200:
                break

            data = response.json()
            batch = data.get("businesses", [])

            if not batch:
                break

            for b in batch:
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
                term_businesses.append(item)

            offset += 50
            time.sleep(0.5)  # Slightly faster harvest

        except:
            break

    return term_businesses


# --- MAIN EXECUTION ---
master_db = []
seen_ids = set()

print("ORBIT SYSTEM: Initiating MEGA HARVEST SEQUENCE...")
print(f"Target: ~5,000 Locations across {len(SEARCH_TERMS)} sectors.")

total_added = 0

for zone in TARGET_ZONES:
    print(f"\n--- DEPLOYING SCANNERS TO: {zone.upper()} ---")

    for term in SEARCH_TERMS:
        results = harvest_sector(term, zone)

        new_in_sector = 0
        for item in results:
            if item["id"] not in seen_ids:
                master_db.append(item)
                seen_ids.add(item["id"])
                new_in_sector += 1

        if new_in_sector > 0:
            total_added += new_in_sector
            # Minimal logging to keep console clean
            # print(f"      + {new_in_sector} new locations.")

    print(f"Current Total: {len(master_db)} locations.")

# --- SAVE ---
with open(OUTPUT_FILE, "w") as f:
    json.dump(master_db, f, indent=4)

print("\n" + "=" * 40)
print(f"MEGA HARVEST COMPLETE.")
print(f"Total Unique Locations: {len(master_db)}")
print(f"Database saved to '{OUTPUT_FILE}'")
print("=" * 40)
