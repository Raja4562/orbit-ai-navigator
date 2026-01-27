import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ORBIT // NAV_SYSTEM",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CYBERPUNK STYLING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #00FF41; font-family: 'Courier New', monospace; }
    .stTextInput > div > div > input { background-color: #111; color: #00FF41; border: 1px solid #333; }
    .stChatMessage { background-color: #0a0a0a; border-left: 3px solid #00FF41; }
    /* MAP BUTTON STYLE */
    .stButton button {
        background-color: #00FF41;
        color: black;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #00cc33;
        color: black;
    }
    #MainMenu, header, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v2"  # <--- ENSURE THIS IS V2


# --- LOAD BRAIN ---
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
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    print("STATUS: System Ready.")
    return tokenizer, model


try:
    tokenizer, model = load_orbit_brain()
except Exception as e:
    st.error(f"SYSTEM FAILURE: {e}")
    st.stop()


# --- INTELLIGENT CHAT LOGIC ---
def get_orbit_response(current_input, history):
    full_prompt = ""
    for msg in history[-4:]:
        role = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
        full_prompt += f"{role}\n{msg['content']}<|end|>\n"

    full_prompt += f"<|user|>\n{current_input}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            use_cache=False,
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# --- COORDS EXTRACTOR ---
def extract_coords(text):
    # Looks for pattern: "COORDS: 37.123, -121.123"
    match = re.search(r"COORDS:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)", text)
    if match:
        return match.group(1), match.group(2)
    return None, None


# --- THE UI ---
st.title("ORBIT // NAV_SYSTEM_V2")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # If this message has coordinates, show the button
        if message["role"] == "assistant":
            lat, lon = extract_coords(message["content"])
            if lat and lon:
                google_maps_url = (
                    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                )
                st.link_button(f"🗺️ ENGAGE GPS TARGET [{lat}, {lon}]", google_maps_url)

# Handle Input
if prompt := st.chat_input("ENTER COMMAND SEQUENCE..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("CALCULATING TRAJECTORY..."):
            response = get_orbit_response(prompt, st.session_state.messages[:-1])
            st.markdown(response)

            # Auto-Extract Coords for the new message
            lat, lon = extract_coords(response)
            if lat and lon:
                google_maps_url = (
                    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                )
                st.link_button(f"🗺️ ENGAGE GPS TARGET [{lat}, {lon}]", google_maps_url)

    st.session_state.messages.append({"role": "assistant", "content": response})
