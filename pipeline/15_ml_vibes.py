import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
INPUT_FILE = "milpitas_rich.json"
OUTPUT_FILE = "milpitas_ml_vibe.json"

print(f"ORBIT ML: Loading {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    print("CRITICAL ERROR: Input file not found.")
    exit()

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# 1. PREPARE DATA FOR ML
# We cluster based on: Rating, Review Count (Log Scale), and Price
ml_data = []
valid_indices = []

for i, place in enumerate(data):
    # Log transform reviews to handle huge variance (10 vs 5000 reviews)
    log_reviews = np.log1p(place.get("review_count", 0))
    rating = place.get("rating", 0)

    # Encode Price: $ = 1, $$ = 2, $$$ = 3, $$$$ = 4, Missing = 1
    price_str = place.get("price", "$")
    price_val = len(price_str) if price_str and price_str != "?" else 1

    ml_data.append([rating, log_reviews, price_val])
    valid_indices.append(i)

# 2. NORMALIZE DATA (Scale to 0-1 range)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ml_data)

# 3. EXECUTE K-MEANS CLUSTERING
# We ask the AI to find 5 distinct "Types" of businesses
print("ORBIT ML: Running K-Means Algorithm on 22,000 locations...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 4. ANALYZE CLUSTERS (What did the math find?)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = {}

print("\n--- ML CLUSTER ANALYSIS ---")
for i, center in enumerate(cluster_centers):
    avg_rating = center[0]
    avg_reviews = np.expm1(center[1])  # Reverse log to get real number
    avg_price = center[2]

    print(
        f"Cluster {i}: Rating {avg_rating:.1f} | Reviews {int(avg_reviews)} | Price Lvl {avg_price:.1f}"
    )

    # Assign Auto-Names based on the mathematical centers
    if avg_reviews > 800:
        cluster_labels[i] = "Viral Hotspot"
    elif avg_rating > 4.2 and avg_reviews < 150:
        cluster_labels[i] = "Hidden Gem"
    elif avg_price > 2.2:
        cluster_labels[i] = "High End"
    elif avg_rating < 3.2:
        cluster_labels[i] = "Risky"
    else:
        cluster_labels[i] = "Standard"

# 5. INJECT TAGS
print("\nORBIT ML: Injecting AI Tags into Database...")
for idx, cluster_id in zip(valid_indices, clusters):
    tag = cluster_labels[cluster_id]

    if "vibes" not in data[idx]:
        data[idx]["vibes"] = []

    if tag not in data[idx]["vibes"]:
        data[idx]["vibes"].append(tag)

# Save
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"SUCCESS: Intelligence injected. Saved to {OUTPUT_FILE}")
