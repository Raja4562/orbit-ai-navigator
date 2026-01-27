import os

# FORCE CACHE TO D: DRIVE
os.environ["HF_HOME"] = "D:/huggingface_cache"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "orbit_model_v2"  # <--- NEW NAME
DATA_FILE = "milpitas_clean.json"  # <--- NEW CLEAN DATA

print("ORBIT SYSTEM: Initializing V2 Training Sequence...")

# 1. LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("STATUS: Loading Model to GPU 0...")
# Hybrid Mode: Force GPU 0 but keep memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map={"": 0}
)

# --- MEMORY OPTIMIZATIONS ---
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 2. APPLY LORA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. PREPARE DATASET
print("STATUS: Formatting Clean Data...")


def format_data(sample):
    # This format teaches it to give the address properly
    prompt = f"<|user|>\nI am in Milpitas. {sample['category']}: {sample['name']}?<|end|>\n<|assistant|>\nRecommended: {sample['name']}. It is located at {sample['address']}. | COORDS: {sample['coords'][0]}, {sample['coords'][1]}<|end|>"
    return {"text": prompt}


dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(format_data)

# Keep context short (256) for speed
dataset = dataset.map(
    lambda samples: tokenizer(
        samples["text"], truncation=True, max_length=256, padding="max_length"
    ),
    batched=True,
)

# 4. START TRAINING
print("STATUS: Starting Training Run...")
training_args = TrainingArguments(
    output_dir="./orbit_checkpoints_v2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=50,  # 50 steps is enough for this small dataset
    save_strategy="no",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

print("STATUS: Saving Orbit V2 Brain...")
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print(f"SUCCESS: Model saved to folder '{NEW_MODEL_NAME}'")
