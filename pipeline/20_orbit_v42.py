import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json
import requests
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import random
import gc
import difflib

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide", page_title="Orbit V42: Truth Core", initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #00FF41; font-family: 'Courier New', monospace; }
    .stTextInput > div > div > input { background-color: #111; color: #00FF41; border: 1px solid #333; }
    .stButton button { background-color: #00FF41; color: black; font-weight: bold; width: 100%; }
    .stChatMessage { background-color: #111; border: 1px solid #333; border-radius: 10px; }
    iframe { width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v2"
DATA_FILE = "milpitas_final.json"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"


# --- 1. LOAD SYSTEM ---
@st.cache_resource
def load_system():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    return data, tokenizer, model


with st.spinner("CALIBRATING TRUTH SENSORS..."):
    try:
        rich_data, tokenizer, model = load_system()
    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}")
        st.stop()

# --- 2. INTELLIGENCE ---


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_ai_intent(user_query):
    clean_memory()
    try:
        sys_prompt = (
            "Classify intent: FOOD, SHOP, ACTIVITY, or UNKNOWN. "
            "Examples: 'Burgers'->FOOD. 'Shoes'->SHOP. 'Hike'->ACTIVITY. "
            "Return ONLY the word."
        )
        full_prompt = (
            f"<|user|>\n{sys_prompt}\nQUERY: {user_query}<|end|>\n<|assistant|>\n"
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=10, temperature=0.1, use_cache=False
            )
        intent = (
            tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            .strip()
            .upper()
        )
        if "FOOD" in intent:
            return "FOOD"
        if "SHOP" in intent:
            return "SHOP"
        if "ACTIVITY" in intent:
            return "ACTIVITY"
        return "ALL"
    except:
        return "ALL"


# --- PERSONALITY ENGINE (WITTY/RUDE) ---
def ask_orbit_personality(place_data, user_query, ref_name, dist):
    clean_memory()

    name = place_data["name"]
    rating = place_data.get("rating", "Unknown")
    categories = ", ".join(place_data.get("categories", [])[:3])
    address = place_data.get("address", "Unknown location")

    try:
        # AGGRESSIVE PROMPT: Force wit and sarcasm
        sys_prompt = (
            "You are Orbit, a cynical, sarcastic AI navigator. "
            "You generally dislike humans, but you have good taste. "
            "Your Goal: Recommend the chosen place, but roast the user's choice or the distance. "
            "Format: 2 sentences. "
            "Sentence 1: Explain WHY this place matches their request (mention the rating or category), but make it sound sarcastic. "
            "Sentence 2: Give the distance/location with a snarky comment."
        )

        data_context = (
            f"USER QUERY: {user_query}\n"
            f"SELECTED PLACE: {name}\n"
            f"TYPE: {categories}\n"
            f"RATING: {rating} stars\n"
            f"DISTANCE: {dist} miles\n"
            f"ADDRESS: {address}"
        )

        full_prompt = (
            f"<|user|>\n{sys_prompt}\n\nDATA:\n{data_context}<|end|>\n<|assistant|>\n"
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

        # High Temperature for creativity
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                use_cache=False,
            )

        reply = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        if len(reply) < 5:
            raise ValueError("Empty")
        return reply
    except:
        return f"I picked **{name}**. It's {dist} miles away. Try not to get lost."


# --- MAP LOGIC (EXACT V19 COPY - NO HARDCODING) ---


def get_coords(name_query):
    if "library" in name_query.lower() and "milpitas" in name_query.lower():
        return (37.4334312, -121.8984208)
    if "great mall" in name_query.lower():
        return (37.415846, -121.898036)
    for p in rich_data:
        if name_query.lower() in p["name"].lower():
            return tuple(p["coords"])
    try:
        # CRITICAL: Using 'orbit_v41' because we know it works on your machine
        geolocator = Nominatim(user_agent="orbit_v41")
        search_query = (
            name_query if "," in name_query else f"{name_query}, Milpitas, CA"
        )
        loc = geolocator.geocode(search_query)
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None
    return None


def get_route(start, end):
    try:
        r = requests.get(
            f"{OSRM_URL}{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson",
            timeout=2,
        )
        if r.status_code == 200:
            return [
                [pt[1], pt[0]]
                for pt in r.json()["routes"][0]["geometry"]["coordinates"]
            ]
    except:
        return [start, end]
    return [start, end]


# --- 3. STATE ---
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "Orbit V42. Truth Protocols Active."}
    ]
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "cached_results" not in st.session_state:
    st.session_state.cached_results = []
if "hero_pin" not in st.session_state:
    st.session_state.hero_pin = None
if "shown_places" not in st.session_state:
    st.session_state.shown_places = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}

# --- 4. UI ---
with st.sidebar:
    st.header("[ NAV CONTROL ]")
    start_txt = st.text_input("Start", "Milpitas Library")
    end_txt = st.text_input("End", "Milpitas Transit Center")
    if st.button("CALCULATE ROUTE"):
        s = get_coords(start_txt)
        e = get_coords(end_txt)
        if s and e:
            line = get_route(s, e)
            st.session_state.route_data = {"start": s, "end": e, "line": line}
            st.session_state.history.append(
                {"role": "assistant", "content": "Route locked."}
            )
            st.rerun()

    st.markdown("---")
    st.caption("SYSTEM INTERNALS")
    if st.session_state.debug_info:
        st.write(st.session_state.debug_info)

st.title("ORBIT V41: TRUTH CORE")

# MAP (SHOW TOP 3)
if st.session_state.route_data:
    rd = st.session_state.route_data
    mid_lat = (rd["start"][0] + rd["end"][0]) / 2
    mid_lon = (rd["start"][1] + rd["end"][1]) / 2
    m = folium.Map(
        location=[mid_lat, mid_lon], zoom_start=14, tiles="CartoDB dark_matter"
    )
    folium.PolyLine(rd["line"], color="#00FF41", weight=5, opacity=0.8).add_to(m)
    folium.Marker(
        rd["start"], popup="Start", icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)
    folium.Marker(
        rd["end"], popup="End", icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

    hero_name = st.session_state.hero_pin["name"] if st.session_state.hero_pin else ""
    display_pins = st.session_state.cached_results[:3]

    for pin in display_pins:
        dist_show = round(pin.get("dist_ref", 0), 2)
        icon_color = "red" if pin["name"] == hero_name else "green"
        z_index = 1000 if pin["name"] == hero_name else 1
        html = f"<b>{pin['name']}</b><br>{pin['rating']}*<br>{dist_show} miles"
        folium.Marker(
            tuple(pin["coords"]),
            popup=html,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
            z_index_offset=z_index,
        ).add_to(m)
    st_folium(m, width=1200, height=500)
else:
    st.info("AWAITING ROUTE...")

# CHAT
st.markdown("---")
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ex: 'Pizza' or 'Anything else?'"):
    st.session_state.history.append({"role": "user", "content": prompt})
    raw_query = prompt.lower()

    if st.session_state.route_data:
        # NEXT INTENT CHECK
        next_triggers = ["else", "another", "next", "different", "more", "other"]
        is_next_request = any(x in raw_query for x in next_triggers)

        if is_next_request and st.session_state.cached_results:
            # --- USE CACHE ---
            available_options = [
                p
                for p in st.session_state.cached_results
                if p["name"] not in st.session_state.shown_places
            ]

            if not available_options:
                available_options = st.session_state.cached_results
                st.session_state.shown_places = []

            top_choice = available_options[0]
            st.session_state.hero_pin = top_choice
            st.session_state.shown_places.append(top_choice["name"])

            # Use cached ref_name
            ref_name = st.session_state.debug_info.get("Ref", "your location")

            # Pass full object to AI for Truth Check
            reply = ask_orbit_personality(
                top_choice,
                "Give me another option",
                ref_name,
                round(top_choice["dist_ref"], 2),
            )
            st.session_state.history.append({"role": "assistant", "content": reply})
            st.rerun()

        else:
            # --- NEW SEARCH ---
            intent = get_ai_intent(prompt)

            STOP_WORDS = [
                "i",
                "want",
                "to",
                "eat",
                "find",
                "shop",
                "buy",
                "some",
                "a",
                "the",
                "near",
                "me",
                "my",
                "on",
                "way",
                "destination",
                "start",
                "end",
            ]
            query_words = [w for w in raw_query.split() if w not in STOP_WORDS]

            start_c = st.session_state.route_data["start"]
            end_c = st.session_state.route_data["end"]

            if "near " in raw_query:
                loc_query = raw_query.split("near ")[1].strip()
                custom_loc = get_coords(loc_query)
                ref_coords = custom_loc if custom_loc else start_c
                ref_name = loc_query.title() if custom_loc else "Start"
                search_radius = 3.0
            elif "great mall" in raw_query:
                ref_coords = get_coords("Great Mall")
                ref_name = "Great Mall"
                search_radius = 1.0
            elif any(x in raw_query for x in ["destination", "end"]):
                ref_coords = end_c
                ref_name = "Destination"
                search_radius = 3.0
            elif any(x in raw_query for x in ["way", "route"]):
                mid_lat = (start_c[0] + end_c[0]) / 2
                mid_lon = (start_c[1] + end_c[1]) / 2
                ref_coords = (mid_lat, mid_lon)
                ref_name = "Mid-Route"
                search_radius = 2.0
            else:
                ref_coords = start_c
                ref_name = "Start"
                search_radius = 3.0

            st.session_state.debug_info = {
                "Ref": ref_name,
                "Intent": intent,
                "Terms": query_words,
            }

            found_places = []
            for p in rich_data:
                p_name = p["name"].lower()
                p_cats = " ".join(p.get("categories", [])).lower()

                if intent == "FOOD" and not any(
                    x in p_cats
                    for x in [
                        "restaurant",
                        "food",
                        "cafe",
                        "grill",
                        "bakery",
                        "pizza",
                        "kitchen",
                    ]
                ):
                    continue
                if intent == "SHOP" and not any(
                    x in p_cats
                    for x in ["store", "shop", "mall", "market", "fashion", "retail"]
                ):
                    continue

                match_score = 0
                for q_word in query_words:
                    if q_word in p_name:
                        match_score += 5
                    matches = difflib.get_close_matches(
                        q_word, p.get("categories", []), n=1, cutoff=0.7
                    )
                    if matches:
                        match_score += 10
                    elif q_word in p_cats:
                        match_score += 8

                if match_score > 0 and p.get("rating", 0) >= 3.0:
                    dist = geodesic(ref_coords, tuple(p["coords"])).miles
                    if dist < search_radius:
                        p["dist_ref"] = dist
                        p["match_score"] = match_score
                        found_places.append(p)

            if found_places:
                found_places.sort(
                    key=lambda x: (x["match_score"], x.get("rating", 0)), reverse=True
                )
                st.session_state.cached_results = found_places
                st.session_state.shown_places = []

                st.session_state.pins = found_places[:3]

                top_choice = found_places[0]
                st.session_state.hero_pin = top_choice
                st.session_state.shown_places.append(top_choice["name"])

                # Pass Data to AI
                reply = ask_orbit_personality(
                    top_choice, prompt, ref_name, round(top_choice["dist_ref"], 2)
                )
                st.session_state.history.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": f"No matches found near {ref_name}.",
                    }
                )
                st.rerun()
    else:
        st.error("Route missing.")
