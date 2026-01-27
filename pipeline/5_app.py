import flet as ft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- 1. CONFIGURATION ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_MODEL = "orbit_model_v1"

print("ORBIT APP: Starting up...")

# --- 2. LOAD THE BRAIN ---
print("STATUS: Loading Neural Network... (Wait for it)")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True, device_map={"": 0}
)
model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
model.eval()
print("STATUS: System Ready.")


# --- 3. THE CHAT LOGIC ---
def get_orbit_response(user_text):
    prompt = f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, use_cache=False
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("<|assistant|>")[-1].strip()


# --- 4. THE UI (Flet) ---
def main(page: ft.Page):
    page.title = "Orbit V2 - Milpitas Navigator"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    # The Chat History
    chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    # Function to handle sending messages
    def send_click(e):
        if not user_input.value:
            return

        # 1. Show User Message
        user_msg = user_input.value
        chat_list.controls.append(
            ft.Container(
                content=ft.Text(f"You: {user_msg}", size=16),
                bgcolor="#263238",
                padding=10,
                border_radius=10,
                alignment=ft.Alignment(1, 0),  # Right align
            )
        )

        # Lock input
        user_input.disabled = True
        page.update()

        # 2. Get AI Response
        response = get_orbit_response(user_msg)

        # 3. Show AI Message
        chat_list.controls.append(
            ft.Container(
                content=ft.Text(f"Orbit: {response}", size=16, color="#66bb6a"),
                bgcolor="black",
                padding=10,
                border_radius=10,
                border=ft.Border(
                    top=ft.BorderSide(1, "#66bb6a"),
                    bottom=ft.BorderSide(1, "#66bb6a"),
                    left=ft.BorderSide(1, "#66bb6a"),
                    right=ft.BorderSide(1, "#66bb6a"),
                ),
                alignment=ft.Alignment(-1, 0),  # Left align
            )
        )

        # Unlock input
        user_input.value = ""
        user_input.disabled = False
        user_input.focus()
        page.update()

    # Input Field
    user_input = ft.TextField(
        hint_text="Ask Orbit about Milpitas...",
        expand=True,
        on_submit=send_click,
        border_color="#66bb6a",
    )

    # --- THE CUSTOM BUTTON (Nuclear-Proof) ---
    # This is not a "Button" class. It is a raw Container.
    # It cannot be deprecated.
    send_btn = ft.Container(
        content=ft.Text("SEND", color="black", weight="bold"),
        bgcolor="#66bb6a",
        padding=15,
        border_radius=30,
        on_click=send_click,  # This makes it clickable
        alignment=ft.Alignment(0, 0),  # Center the text
    )

    # Layout
    page.add(
        ft.Text("ORBIT V2", size=30, weight="bold", color="#66bb6a"),
        ft.Divider(color="green"),
        chat_list,
        ft.Row([user_input, send_btn]),
    )


# Run the App
ft.app(target=main)
