import json
from geopy.geocoders import Nominatim
import time
import os

# Initialize Geocoder
geolocator = Nominatim(user_agent="orbit_milpitas_project_v1")

print("ORBIT DATA: Analyzing Coordinates...")

# Load your raw data
input_file = "milpitas_raw.json"
if not os.path.exists(input_file):
    print(f"ERROR: Could not find {input_file}")
    exit()

with open(input_file, "r") as f:
    data = json.load(f)

fixed_data = []

for place in data:
    # Check if address is missing or "NaN"
    address_val = str(place.get("address", "")).lower()

    # "nan" string or empty value needs fixing
    if address_val == "nan" or address_val == "none" or address_val == "":
        lat, lon = place["coords"]
        print(f"Fixing: {place['name']}...")

        try:
            # Ask OpenStreetMap for the address
            location = geolocator.reverse(f"{lat}, {lon}")
            if location:
                place["address"] = location.address
                print(f"   -> FOUND: {location.address[:40]}...")
            else:
                place["address"] = "Milpitas, CA"

            # Sleep to respect API limits
            time.sleep(1.2)

        except Exception as e:
            print(f"   -> ERROR: {e}")
            place["address"] = "Milpitas, CA"

    fixed_data.append(place)

# Save the polished version
with open("milpitas_clean.json", "w") as f:
    json.dump(fixed_data, f, indent=4)

print("\nSUCCESS: Created 'milpitas_clean.json' with real addresses.")
