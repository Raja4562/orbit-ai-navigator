import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
import math
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ORBITD",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CYBERPUNK STYLING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #00FF41; font-family: 'Courier New', monospace; }
    .stTextInput > div > div > input { background-color: #111; color: #00FF41; border: 1px solid #333; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #333; }
    .stButton button { background-color: #00FF41; color: black; font-weight: bold; border: none; width: 100%; }
    .stButton button:hover { background-color: #00cc33; color: black; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #111; color: #00FF41; }
    .stSlider > div > div > div { background-color: #00FF41 !important; }
    /* Minimal Chat Input Styling */
    .stChatInput > div > div > textarea { background-color: #111; color: #00FF41; border: 1px solid #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v2"
DATA_FILE = "milpitas_rich.json"


# --- 1. LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return []


# --- 2. LOAD BRAIN ---
@st.cache_resource
def load_brain():
    print("ORBIT SYSTEM: Loading Neural Network...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    base.config.use_cache = False
    base.generation_config.use_cache = False
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = load_brain()
    rich_data = load_data()
except Exception as e:
    st.error(f"SYSTEM FAILURE: {e}")
    st.stop()


# --- 3. AI ENGINE ---
def ask_orbit(prompt_text):
    full_prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            use_cache=False,
        )
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# --- 4. ENGINE ---
def get_coords(name):
    if "library" in name.lower():
        return (37.4334312, -121.8984208)
    if "great mall" in name.lower():
        return (37.415846, -121.898036)
    geolocator = Nominatim(user_agent="orbit_v4_visual")
    try:
        loc = geolocator.geocode(f"{name}, Milpitas, CA")
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None
    return None


def find_best_stops(start, end, query, min_rating=0, selected_prices=None):
    base_dist = geodesic(start, end).miles
    max_dist = base_dist * 1.5
    candidates = []
    # If chat input is something like "Find Tacos", we just want "Tacos"
    clean_query = query.lower().replace("find", "").replace("show me", "").strip()
    query_terms = clean_query.split()

    for p in rich_data:
        text_match = any(t in p["name"].lower() for t in query_terms) or any(
            t in c.lower() for c in p["categories"] for t in query_terms
        )
        rating_match = p["rating"] >= min_rating
        price_match = True
        if selected_prices and "Any" not in selected_prices:
            if p["price"] not in selected_prices:
                price_match = False

        if text_match and rating_match and price_match:
            try:
                p_coords = tuple(p["coords"])
                trip = geodesic(start, p_coords).miles + geodesic(p_coords, end).miles
                if trip <= max_dist:
                    score = p["rating"] * math.log10(p["review_count"] + 1)
                    p["score"] = score
                    candidates.append(p)
            except:
                continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]


# ==========================================
#               THE UI
# ==========================================

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("MISSION CONTROL")
    start_txt = st.text_input("START", "Milpitas Library")
    end_txt = st.text_input("DESTINATION", "Great Mall")

    # This box now updates if you type in chat
    if "current_search" not in st.session_state:
        st.session_state.current_search = "Pizza"
    query_txt = st.text_input("LOOKING FOR?", st.session_state.current_search)

    st.markdown("---")
    min_rating = st.slider("MIN RATING", 0.0, 5.0, 3.5, 0.5)
    price_opts = ["$", "$$", "$$$", "$$$$"]
    selected_prices = st.multiselect("PRICE", price_opts, default=["$", "$$"])

    run_btn = st.button("CALCULATE & VISUALIZE")

# --- MAIN LAYOUT ---
st.title("ORBIT // HYBRID_COMMAND_V4.4")

if "map_data" not in st.session_state:
    st.session_state.map_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- UNIVERSAL LOGIC HANDLER ---
def execute_search(search_term):
    # Update the sidebar state visually
    st.session_state.current_search = search_term

    with st.spinner(f"SCANNING FOR '{search_term}'..."):
        c_start = get_coords(start_txt)
        c_end = get_coords(end_txt)

        if c_start and c_end:
            results = find_best_stops(
                c_start, c_end, search_term, min_rating, selected_prices
            )

            if results:
                st.session_state.map_data = {
                    "start": c_start,
                    "end": c_end,
                    "stops": results,
                }

                # Generate AI Response
                names = ", ".join([f"{r['name']} ({r['rating']}★)" for r in results])
                ai_prompt = f"User searched for '{search_term}'. Found stops: {names}. Recommend the best one."
                resp = ask_orbit(ai_prompt)

                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                st.warning(f"No results for '{search_term}'. Try adjusting filters.")


# TRIGGER 1: Sidebar Button
if run_btn:
    execute_search(query_txt)

# TRIGGER 2: Chat Input (Minimal & Expandable)
if prompt := st.chat_input("Enter command (e.g., 'Find Burgers', 'Gas Stations')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    execute_search(prompt)
    st.rerun()

# --- MAP RENDERER ---
if st.session_state.map_data:
    md = st.session_state.map_data
    mid_lat = (md["start"][0] + md["end"][0]) / 2
    mid_lon = (md["start"][1] + md["end"][1]) / 2

    m = folium.Map(
        location=[mid_lat, mid_lon], zoom_start=14, tiles="CartoDB dark_matter"
    )
    folium.PolyLine(
        [md["start"], md["end"]], color="#00FF41", weight=3, opacity=0.7
    ).add_to(m)
    folium.Marker(
        md["start"], popup="START", icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)
    folium.Marker(
        md["end"], popup="DESTINATION", icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

    for stop in md["stops"]:
        html = f"<div style='font-family: monospace; width: 150px;'><b>{stop['name']}</b><br>⭐ {stop['rating']} | 💲 {stop['price']}</div>"
        folium.Marker(
            stop["coords"],
            popup=html,
            tooltip=stop["name"],
            icon=folium.Icon(color="green", icon="star"),
        ).add_to(m)

    st_folium(m, width=1200, height=500)

# --- CHAT HISTORY (Appears below map) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
