import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

model_dir = "/content/drive/MyDrive/GoogleColab/AI/Qwen3/models/Qwen3-4B-Base"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print("Base model config:")
print(model.config)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, peft_config)

param_counts = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        # get module type / name
        mod_type = name.split(".")[-2] if "." in name else name
        param_counts[mod_type] = param_counts.get(mod_type, 0) + p.numel()

print("\nTrainable parameters per module type:")
for k, v in sorted(param_counts.items()):
    print(f"  {k:15s}: {v:,}")

total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable: {total_trainable:,}")
print(f"Total params:    {total_params:,}")
print(f"Percentage:      {(total_trainable/total_params)*100:.4f}%")
