import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json
import folium
from streamlit_folium import st_folium
import requests
from geopy.geocoders import Nominatim

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="ORBIT V17: NAVIGATOR", layout="wide")
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v2"
DATA_FILE = "milpitas_final.json"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

# Custom CSS for that "Orbit" feel
st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; color: #00FF41; }
    .stChatInput { position: fixed; bottom: 30px; }
    div[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #333; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 2. THE BRAIN (AI) ---
@st.cache_resource
def load_brain():
    # Only load if not already loaded (prevents reloading lag)
    try:
        print("ORBIT: Waking up...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Brain Damage: {e}")
        return None, None


@st.cache_resource
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return []


tokenizer, model = load_brain()
rich_data = load_data()


def ask_orbit_witty(context, user_query):
    """
    This function forces the AI to be witty and HIDES the coordinates.
    It takes the search results (context) and generates a human response.
    """
    sys_prompt = (
        "You are Orbit, a sarcastic, witty AI navigator. "
        "You are helping a user drive. "
        "Do NOT mention latitude/longitude coordinates. "
        "Do NOT sound robotic. "
        "If recommending a place, give a short, punchy reason why it's good."
    )

    full_prompt = f"<|user|>\n{sys_prompt}\nDATA FOUND: {context}\nUSER ASKED: {user_query}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, temperature=0.8, use_cache=False
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


# --- 3. THE MAP ENGINE ---
def get_coords(name):
    # Hardcoded fixes for stability
    if "library" in name.lower():
        return (37.4334, -121.8984)
    if "transit" in name.lower():
        return (37.435, -121.895)
    if "great mall" in name.lower():
        return (37.4158, -121.8980)

    # Search Data
    for p in rich_data:
        if name.lower() in p["name"].lower():
            return tuple(p["coords"])

    # Fallback
    try:
        geo = Nominatim(user_agent="orbit_v17").geocode(f"{name}, Milpitas, CA")
        if geo:
            return (geo.latitude, geo.longitude)
    except:
        return None
    return None


def get_route(c1, c2):
    try:
        r = requests.get(
            f"{OSRM_URL}{c1[1]},{c1[0]};{c2[1]},{c2[0]}?overview=full&geometries=geojson"
        )
        return [
            [pt[1], pt[0]] for pt in r.json()["routes"][0]["geometry"]["coordinates"]
        ]
    except:
        return [c1, c2]  # Straight line fallback


# --- 4. SESSION STATE (The Memory) ---
if "route" not in st.session_state:
    st.session_state.route = None
if "pins" not in st.session_state:
    st.session_state.pins = []  # For burgers, etc.
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "assistant",
            "content": "Orbit Online. Set your route on the left, then we can talk.",
        }
    ]

# --- 5. SIDEBAR: NAVIGATION CONTROLS ---
with st.sidebar:
    st.header("📍 NAVIGATION DECK")
    start_txt = st.text_input("Start Point", "Milpitas Library")
    end_txt = st.text_input("End Point", "Milpitas Transit Center")

    if st.button("START ENGINE"):
        c1 = get_coords(start_txt)
        c2 = get_coords(end_txt)
        if c1 and c2:
            st.session_state.route = get_route(c1, c2)
            st.session_state.pins = []  # Clear old food pins
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": "Route calculated. I'm bored though. Want to stop for food or something?",
                }
            )
            st.rerun()
        else:
            st.error("Could not find those locations.")

# --- 6. MAIN UI: MAP & CHAT ---

# A. THE MAP (Stable, only updates on state change)
st.markdown("### 🗺️ LIVE FEED")
if st.session_state.route:
    # Center map between start and end
    mid_lat = (st.session_state.route[0][0] + st.session_state.route[-1][0]) / 2
    mid_lon = (st.session_state.route[0][1] + st.session_state.route[-1][1]) / 2

    m = folium.Map(
        location=[mid_lat, mid_lon], zoom_start=14, tiles="CartoDB dark_matter"
    )

    # Draw Route
    folium.PolyLine(
        st.session_state.route, color="#00FF41", weight=5, opacity=0.8
    ).add_to(m)
    folium.Marker(
        st.session_state.route[0],
        popup="Start",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(m)
    folium.Marker(
        st.session_state.route[-1],
        popup="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Draw "Burger" Pins (if any)
    for pin in st.session_state.pins:
        folium.Marker(
            pin["coords"],
            popup=f"<b>{pin['name']}</b><br>{pin['rating']}⭐",
            icon=folium.Icon(color="orange", icon="cutlery"),
        ).add_to(m)

    st_folium(m, width=1200, height=400)
else:
    st.info("SYSTEM IDLE. Set coordinates in sidebar.")


# B. THE CHAT (Witty AI)
st.markdown("---")
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Talk to Orbit..."):
    st.session_state.history.append({"role": "user", "content": prompt})

    # LOGIC: SEARCH vs CHAT
    search_term = prompt.lower().replace("i want", "").replace("find", "").strip()

    # 1. Search Logic (Python finds the data)
    found_items = []
    if rich_data:
        for p in rich_data:
            # Match keywords (burger, pizza, etc)
            blob = (
                p["name"]
                + " "
                + " ".join(p.get("categories", []))
                + " "
                + " ".join(p.get("vibes", []))
            ).lower()
            if search_term in blob or (search_term in "hungry" and "food" in blob):
                if p.get("rating", 0) > 4.0:  # Only good stuff
                    found_items.append(p)

    # 2. Update Map State (The "Pins")
    if found_items:
        top_5 = found_items[:5]
        st.session_state.pins = top_5  # This puts them on the map next refresh

        # 3. Generate Witty Response (AI writes the text)
        context_str = ", ".join(
            [
                f"{p['name']} ({p['rating']} stars, vibe: {p.get('vibes',['nice'])[0]})"
                for p in top_5
            ]
        )
        orbit_reply = ask_orbit_witty(context_str, prompt)

        st.session_state.history.append({"role": "assistant", "content": orbit_reply})
        st.rerun()  # Refresh to show new pins

    else:
        # Just Chatting
        orbit_reply = ask_orbit_witty(
            "No specific map data found for this query.", prompt
        )
        st.session_state.history.append({"role": "assistant", "content": orbit_reply})
        st.rerun()
