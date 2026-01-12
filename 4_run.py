import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v1"

print("ORBIT SYSTEM: Loading Neural Network...")

# 1. LOAD BASE BRAIN
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True, device_map={"": 0}
)

# 2. LOAD YOUR CUSTOM MILPITAS KNOWLEDGE
print("STATUS: Injecting Milpitas Knowledge...")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()


# 3. DEFINE THE TEST
def ask_orbit(question, coords="37.4323, -121.8996"):
    prompt = f"<|user|>\nI am at {coords}. {question}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # --- THE FIX IS HERE ---
    # We add use_cache=False to prevent the 'seen_tokens' crash
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        use_cache=False,  # <--- CRITICAL FIX
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()


# 4. RUN TESTS
print("\n" + "=" * 50)
print("ORBIT LIVE TEST")
print("=" * 50)

q1 = "I am hungry. Suggest a place."
print(f"\nUser: {q1}")
print(f"Orbit: {ask_orbit(q1)}")

q2 = "Where can I go for a walk?"
print(f"\nUser: {q2}")
print(f"Orbit: {ask_orbit(q2)}")

print("\n" + "=" * 50)
