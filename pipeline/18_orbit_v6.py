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

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="Orbit V23: Personality Core",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #00FF41; font-family: 'Courier New', monospace; }
    .stTextInput > div > div > input { background-color: #111; color: #00FF41; border: 1px solid #333; }
    .stButton button { background-color: #00FF41; color: black; font-weight: bold; width: 100%; }
    /* Chat Bubbles */
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


with st.spinner("INJECTING PERSONALITY MATRIX..."):
    try:
        rich_data, tokenizer, model = load_system()
    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}")
        st.stop()


# --- 2. INTELLIGENCE (THE NEW PERSONALITY) ---
def ask_orbit_personality(context, user_query, vibe):
    # This prompt is aggressive to force "Wit"
    sys_prompt = (
        "You are Orbit, a cynical, sarcastic, cyberpunk AI navigator. "
        "You do not give boring directions. You give opinions. "
        "If the place is 4+ stars, hype it up but warn about crowds. "
        "If it's < 4 stars, be skeptical. "
        f"The vibe of this place is: {vibe}. Use that in your joke. "
        "Keep it under 3 sentences. Make the user laugh."
    )

    full_prompt = f"<|user|>\n{sys_prompt}\nPLACE DETAILS: {context}\nUSER ASKED: {user_query}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # High Temp = More Creative/Witty
        outputs = model.generate(
            **inputs, max_new_tokens=200, temperature=0.9, use_cache=False
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def get_coords(name):
    if "library" in name.lower():
        return (37.4334312, -121.8984208)
    if "transit" in name.lower():
        return (37.4103, -121.8910)
    for p in rich_data:
        if name.lower() in p["name"].lower():
            return tuple(p["coords"])
    try:
        loc = Nominatim(user_agent="orbit_v23").geocode(f"{name}, Milpitas, CA")
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
        {
            "role": "assistant",
            "content": "Orbit V23 Online. Personality module loaded. Try me.",
        }
    ]
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "pins" not in st.session_state:
    st.session_state.pins = []
if "shown_places" not in st.session_state:
    st.session_state.shown_places = []
if "last_search_term" not in st.session_state:
    st.session_state.last_search_term = None

# --- 4. UI ---
with st.sidebar:
    st.header("NAV CONTROL")
    start_txt = st.text_input("Start", "Milpitas Library")
    end_txt = st.text_input("End", "Milpitas Transit Center")
    if st.button("CALCULATE ROUTE"):
        s = get_coords(start_txt)
        e = get_coords(end_txt)
        if s and e:
            line = get_route(s, e)
            st.session_state.route_data = {"start": s, "end": e, "line": line}
            st.session_state.pins = []
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": "Route locked. I'm bored. Let's make a stop.",
                }
            )
            st.rerun()

st.title("ORBIT V23: AI CORE")

# MAP
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

    for pin in st.session_state.pins:
        dist_show = round(pin.get("dist_from_start", 0), 2)
        vibe_tag = pin.get("vibes", ["Standard"])[0]
        html = f"<b>{pin['name']}</b><br>{pin['rating']}⭐<br>Vibe: {vibe_tag}"
        folium.Marker(
            tuple(pin["coords"]),
            popup=html,
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(m)
    st_folium(m, width=1200, height=500)
else:
    st.info("AWAITING ROUTE DATA...")

# CHAT
st.markdown("---")
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ex: 'I want burgers' or 'Anything else?'"):
    st.session_state.history.append({"role": "user", "content": prompt})

    raw_query = prompt.lower()

    # 1. KEYWORD EXTRACTION
    keywords = [
        "pizza",
        "burger",
        "sushi",
        "taco",
        "coffee",
        "tea",
        "chinese",
        "mexican",
        "indian",
        "italian",
        "thai",
    ]
    current_keyword = next((word for word in keywords if word in raw_query), None)

    if "buger" in raw_query:
        current_keyword = "burger"

    # 2. MEMORY LOGIC
    if (
        not current_keyword
        and st.session_state.last_search_term
        and any(x in raw_query for x in ["else", "another", "more", "different"])
    ):
        search_term = st.session_state.last_search_term
        use_memory = True
    elif current_keyword:
        search_term = current_keyword
        st.session_state.last_search_term = search_term
        st.session_state.shown_places = []  # Reset on new topic
        use_memory = False
    else:
        search_term = raw_query.replace("find", "").replace("i want", "").strip()
        use_memory = False

    # 3. SEARCH
    found_places = []
    if st.session_state.route_data:
        start_coords = st.session_state.route_data["start"]

        for p in rich_data:
            match = (
                search_term in p["name"].lower()
                or any(search_term in c.lower() for c in p.get("categories", []))
                or any(search_term in v.lower() for v in p.get("vibes", []))
            )

            if match and p.get("rating", 0) >= 3.5:
                # DISTANCE FILTER
                dist = geodesic(start_coords, tuple(p["coords"])).miles
                if dist < 4.0:
                    p["dist_from_start"] = dist
                    found_places.append(p)
    else:
        st.error("Route missing.")

    # 4. PERSONALITY GENERATION
    if found_places:
        found_places.sort(key=lambda x: x.get("rating", 0), reverse=True)

        # Filter ALREADY shown
        new_options = [
            p for p in found_places if p["name"] not in st.session_state.shown_places
        ]
        if not new_options:
            new_options = found_places  # Cycle back if empty

        top_choice = new_options[0]
        st.session_state.shown_places.append(top_choice["name"])
        st.session_state.pins = [top_choice]

        # Prepare Data for AI
        dist = round(top_choice["dist_from_start"], 2)
        vibe = top_choice.get("vibes", ["Standard"])[0]
        context = (
            f"{top_choice['name']} ({top_choice['rating']} stars, {dist} miles away)."
        )

        # THE MAGIC CALL
        reply = ask_orbit_personality(context, prompt, vibe)

        st.session_state.history.append({"role": "assistant", "content": reply})
        st.rerun()

    elif "hungry" in raw_query:
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "I'm not a mind reader. Tell me what you crave. Burgers? Tacos? Space food?",
            }
        )
        st.rerun()
    else:
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "Zero matches. Milpitas is a food desert for that specific request.",
            }
        )
        st.rerun()
