import json
import random

# --- CONFIGURATION ---
INPUT_FILE = "milpitas_raw.json"
OUTPUT_FILE = "milpitas_training_data.jsonl"

# --- THE "BUTLER" TEMPLATES ---
# This simulates how we WANT the AI to talk.
# We create 3 variations for every place.

prompts_nature = [
    "I need a quiet place to think.",
    "Where can I see some green?",
    "I want to go for a walk away from cars.",
    "Suggest a nature spot.",
    "I am stressed, I need a view.",
]

prompts_food = [
    "I am hungry.",
    "Where is a good place to eat?",
    "I want food nearby.",
    "Suggest a restaurant.",
    "I need dinner recommendations.",
]

prompts_shop = [
    "I need to buy something.",
    "Where is the nearest mall?",
    "I want to go shopping.",
    "Is there a store around here?",
    "Shopping recommendation.",
]

print("ORBIT SYSTEM: Synthesizing AI Training Data...")

try:
    with open(INPUT_FILE, "r") as f:
        raw_data = json.load(f)

    training_data = []

    for place in raw_data:
        name = place["name"]
        addr = place["address"]
        cat = place["category"]
        lat, lon = place["coords"]

        # 1. CREATE "BUTLER" RESPONSE
        # The AI should sound polite and helpful.
        if "park" in cat or "nature" in cat:
            user_input = random.choice(prompts_nature)
            response = f"Sir, I recommend {name}. It is a peaceful location nearby. Perfect for clearing your mind."
        elif "shop" in cat or "mall" in cat:
            user_input = random.choice(prompts_shop)
            response = f"You are close to {name}. It is excellent for shopping. I have set the destination."
        else:  # Food/Other
            user_input = random.choice(prompts_food)
            response = f"If you are hungry, {name} is an excellent choice. It is located at {addr}. Shall we go?"

        # 2. FORMAT FOR Llama 3 / Mistral (Alpaca Format)
        # This is the standard format for fine-tuning.
        entry = {
            "instruction": f"You are Orbit, a helpful navigation assistant for Milpitas. User Location: {lat:.4f}, {lon:.4f}",
            "input": user_input,
            "output": f"{response} | COORDS: {lat}, {lon}",
        }

        training_data.append(entry)

        # 3. ADD A "DIRECT" VARIATION (Just the facts)
        entry_direct = {
            "instruction": f"Navigate to {name}.",
            "input": "Where is it?",
            "output": f"{name} is located at {addr}. I am calculating the route now. | COORDS: {lat}, {lon}",
        }
        training_data.append(entry_direct)

    # SAVE AS JSONL (JSON Lines)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(training_data, f, indent=4)

    print(f"SUCCESS: Generated {len(training_data)} training examples.")
    print(f"Saved to '{OUTPUT_FILE}'. ready for the Brain.")

except Exception as e:
    print(f"ERROR: {e}")
