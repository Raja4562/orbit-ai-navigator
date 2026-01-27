import osmnx as ox
import pandas as pd
import json

# --- CONFIGURATION ---
CITY = "Milpitas, California, USA"
TAGS = {
    "amenity": ["restaurant", "cafe", "fast_food", "pub", "bar", "library", "cinema"],
    "leisure": ["park", "garden", "nature_reserve"],
    "shop": ["mall", "department_store", "supermarket", "clothes"],
}

print(f"ORBIT SYSTEM: Scanning {CITY} for data points...")

try:
    # 1. DOWNLOAD RAW DATA
    pois = ox.features_from_place(CITY, tags=TAGS)
    print(f"STATUS: Found {len(pois)} raw locations.")

    # 2. CLEAN DATA
    orbit_data = []
    for index, row in pois.iterrows():
        # Only keep places with a name
        if "name" in row and pd.notna(row["name"]):

            # Determine Category
            category = "Unknown"
            if "amenity" in row and pd.notna(row["amenity"]):
                category = row["amenity"]
            elif "leisure" in row and pd.notna(row["leisure"]):
                category = row["leisure"]
            elif "shop" in row and pd.notna(row["shop"]):
                category = row["shop"]

            # Get Coordinates
            if hasattr(row.geometry, "centroid"):
                lat = row.geometry.centroid.y
                lon = row.geometry.centroid.x
            else:
                lat = row.geometry.y
                lon = row.geometry.x

            entry = {
                "name": row["name"],
                "category": category,
                "address": row.get("addr:street", "Milpitas, CA"),
                "coords": [lat, lon],
            }
            orbit_data.append(entry)

    # 3. SAVE TO FILE
    with open("milpitas_raw.json", "w") as f:
        json.dump(orbit_data, f, indent=4)

    print(f"SUCCESS: Saved {len(orbit_data)} clean locations to 'milpitas_raw.json'")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
