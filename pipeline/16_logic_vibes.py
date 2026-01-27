import json
import os

# --- CONFIGURATION ---
INPUT_FILE = "milpitas_ml_vibe.json"
OUTPUT_FILE = "milpitas_final.json"

print(f"ORBIT LOGIC: Loading {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    print("CRITICAL ERROR: Input file not found.")
    exit()

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

print("ORBIT LOGIC: Applying Vibe Algorithms...")

count = 0
for place in data:
    # Ensure vibes list exists (it should from ML step)
    if "vibes" not in place:
        place["vibes"] = []

    # Get data for logic
    cats = [c.lower() for c in place.get("categories", [])]
    price = place.get("price", "?")
    rating = place.get("rating", 0)

    # --- LOGIC RULES (The "Synthetic Review" Source) ---

    # VIBE: DATE NIGHT 🌙
    # Logic: Expensive + Romantic Cuisine
    romantic_food = ["italian", "steak", "french", "wine", "sushi", "tapas", "fusion"]
    if (price == "$$$" or price == "$$$$") and any(
        f in c for c in cats for f in romantic_food
    ):
        place["vibes"].append("Date Night")

    # VIBE: STUDY SPOT 💻
    # Logic: Coffee/Tea + Cheap
    is_cafe = any(c in ["coffee", "tea", "bubble tea", "juice", "cafe"] for c in cats)
    if is_cafe and price == "$":
        place["vibes"].append("Study Spot")

    # VIBE: LATE NIGHT 🦉
    # Logic: Bars, Karaoke
    night_cats = ["bar", "pub", "karaoke", "hookah", "nightlife", "club"]
    if any(n in c for c in cats for n in night_cats):
        place["vibes"].append("Late Night")

    # VIBE: BUDGET EATS 💸
    # Logic: Cheap + Good Rating
    if price == "$" and rating >= 4.0:
        place["vibes"].append("Budget Eats")

    # VIBE: GROUP HANG 🎉
    # Logic: Pizza, KBBQ, Hot Pot (Social Foods)
    social_food = ["pizza", "barbeque", "hot pot", "bowling", "arcade"]
    if any(s in c for c in cats for s in social_food):
        place["vibes"].append("Group Hang")

    # Clean up duplicates
    place["vibes"] = list(set(place["vibes"]))
    count += 1

# Save Final Database
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"SUCCESS: Logic applied to {count} locations.")
print(f"FINAL DATABASE READY: {OUTPUT_FILE}")
