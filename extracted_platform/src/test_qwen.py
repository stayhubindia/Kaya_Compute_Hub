import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model in 4-bit...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
)

print("\nModel loaded successfully!")
print("GPU:", torch.cuda.get_device_name(0))
print("Allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")

prompt = "Explain what Linux is in simple words."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )

print("\n--- RESPONSE ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
