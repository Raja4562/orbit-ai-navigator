import json
from geopy.distance import geodesic

# --- CONFIGURATION ---
# Let's pretend we are driving from the Library to the Great Mall
START_POINT = (37.4334312, -121.8984208)  # Milpitas Library
END_POINT = (37.415846, -121.898036)  # Great Mall

# Maximum allowed detour (1.3 = trip can be 30% longer max)
DETOUR_TOLERANCE = 1.3


def load_data():
    with open("milpitas_clean.json", "r") as f:
        return json.load(f)


def find_detours(start, end, data):
    print(f"--- DETOUR CALCULATION ENGINE ---")
    print(f"Start: {start}")
    print(f"End:   {end}")

    # 1. Calculate Baseline Distance (Direct Route)
    # geodesic gives us miles/km between two coords
    base_dist = geodesic(start, end).miles
    print(f"Direct Distance: {base_dist:.2f} miles")
    print(f"Max Allowed Trip: {base_dist * DETOUR_TOLERANCE:.2f} miles")
    print("-" * 40)

    valid_places = []

    for place in data:
        place_coords = tuple(place["coords"])  # Needs to be (lat, lon)

        # 2. Calculate the "Dog Leg" (Start -> Place -> End)
        dist_to_place = geodesic(start, place_coords).miles
        dist_from_place = geodesic(place_coords, end).miles
        total_trip = dist_to_place + dist_from_place

        # 3. Calculate "Detour Cost" (How many extra miles?)
        extra_miles = total_trip - base_dist

        # 4. Filter
        if total_trip <= (base_dist * DETOUR_TOLERANCE):
            # It's a valid detour!
            valid_places.append(
                {
                    "name": place["name"],
                    "category": place["category"],
                    "address": place["address"],
                    "extra_miles": extra_miles,
                    "total_trip": total_trip,
                }
            )

    # Sort by "least extra driving" (Efficiency)
    valid_places.sort(key=lambda x: x["extra_miles"])
    return valid_places


# --- RUN THE ENGINE ---
data = load_data()
detours = find_detours(START_POINT, END_POINT, data)

print(f"FOUND {len(detours)} VALID STOPS ON THE WAY:\n")

for i, p in enumerate(detours[:5]):  # Show top 5
    print(f"{i+1}. {p['name']} ({p['category']})")
    print(f"   Address: {p['address']}")
    print(f"   Detour Cost: +{p['extra_miles']:.2f} miles extra")
    print("-" * 30)
