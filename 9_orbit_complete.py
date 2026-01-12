import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import json
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ORBIT // NAV_SYSTEM_V3",
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
    .stChatMessage { background-color: #0a0a0a; border-left: 3px solid #00FF41; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #333; }
    .stButton button { background-color: #00FF41; color: black; font-weight: bold; border: none; width: 100%; }
    .stButton button:hover { background-color: #00cc33; color: black; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v2"


# --- 1. LOAD BRAIN ---
@st.cache_resource
def load_orbit_brain():
    print("ORBIT SYSTEM: Loading Neural Network...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    # Fix for crash
    base.config.use_cache = False
    base.generation_config.use_cache = False

    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    return tokenizer, model


# --- 2. LOAD MAP DATA ---
@st.cache_resource
def load_map_data():
    try:
        with open("milpitas_clean.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# --- 3. DETOUR MATH ENGINE ---
def find_detours(start_coords, end_coords, data):
    try:
        base_dist = geodesic(start_coords, end_coords).miles
        max_dist = base_dist * 1.3

        valid_places = []
        for place in data:
            try:
                p_coords = tuple(place["coords"])
                trip_dist = (
                    geodesic(start_coords, p_coords).miles
                    + geodesic(p_coords, end_coords).miles
                )
                if trip_dist <= max_dist:
                    extra = trip_dist - base_dist
                    valid_places.append(
                        {
                            "name": place["name"],
                            "coords": place["coords"],
                            "extra": extra,
                        }
                    )
            except:
                continue
        valid_places.sort(key=lambda x: x["extra"])
        return valid_places[:5]
    except:
        return []


# --- 4. GEOCODING ---
def get_coords_from_name(name):
    if "," in name and any(c.isdigit() for c in name):
        try:
            parts = name.split(",")
            return float(parts[0]), float(parts[1])
        except:
            pass
    geolocator = Nominatim(user_agent="orbit_v3_integrated")
    try:
        loc = geolocator.geocode(f"{name}, Milpitas, CA")
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None
    return None


# --- LOAD RESOURCES ---
try:
    tokenizer, model = load_orbit_brain()
    map_data = load_map_data()
except Exception as e:
    st.error(f"SYSTEM FAILURE: {e}")
    st.stop()


# --- AI GENERATION (FIXED NO ECHO) ---
def ask_orbit(prompt_text):
    full_prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            use_cache=False,
        )

    # TOKEN SLICING: Completely removes the echo
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return resp.strip()


# ==========================================
#               THE UI LAYOUT
# ==========================================
with st.sidebar:
    st.title("MISSION CONTROL")
    st.markdown("---")
    start_loc = st.text_input("START POINT", "Milpitas Library")
    end_loc = st.text_input("DESTINATION", "Great Mall")

    if st.button("CALCULATE ROUTE"):
        with st.spinner("COMPUTING GEOMETRY..."):
            c_start = get_coords_from_name(start_loc)
            c_end = get_coords_from_name(end_loc)
            if c_start and c_end:
                candidates = find_detours(c_start, c_end, map_data)
                if candidates:
                    st.success(f"FOUND {len(candidates)} OPTIONS")
                    st.session_state["route_candidates"] = candidates
                else:
                    st.warning("NO DETOURS FOUND")
            else:
                st.error("COULD NOT FIND LOCATIONS")

st.title("ORBIT // NAV_SYSTEM_V3")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "coords" in msg:
            lat, lon = msg["coords"]
            url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            st.link_button(f"🗺️ ENGAGE GPS [{lat}, {lon}]", url)

if "route_candidates" in st.session_state:
    candidates = st.session_state.pop("route_candidates")
    options_text = ", ".join([c["name"] for c in candidates])

    user_msg = f"I am going from {start_loc} to {end_loc}. Analyze these stops: {options_text}. Which is best?"
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # We send the candidates to the AI, but it won't repeat them because of token slicing
    ai_prompt = f"I am travelling from {start_loc} to {end_loc}. The available stops are: {options_text}. Recommend one."
    response = ask_orbit(ai_prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if prompt := st.chat_input("ENTER COMMAND..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("PROCESSING..."):
            resp = ask_orbit(prompt)
            st.markdown(resp)

            match = re.search(r"COORDS:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)", resp)
            coords = None
            if match:
                coords = (match.group(1), match.group(2))
                lat, lon = coords
                url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                st.link_button(f"🗺️ ENGAGE GPS [{lat}, {lon}]", url)

    msg_data = {"role": "assistant", "content": resp}
    if coords:
        msg_data["coords"] = coords
    st.session_state.messages.append(msg_data)
