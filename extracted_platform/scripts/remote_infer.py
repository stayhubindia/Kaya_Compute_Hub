#!/usr/bin/env python3
"""
Remote GPU Inference Script for Colab VM (Phase 5.1).
Loads Qwen3-4B-Base + LoRA Adapter on Tesla T4 GPU in VRAM and responds to prompts.
Optimized for ultra-fast FP16 inference (6GB VRAM footprint on Tesla T4 GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_COLAB_SECRET_ACCESS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

current_file = globals().get("__file__")
PROJECT_ROOT = Path("/content") if Path("/content/outputs").exists() else (Path(current_file).resolve().parent.parent if current_file else Path.cwd())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Prompt text")
    parser.add_argument("--system-prompt", "-s", type=str, default="You are an expert AI assistant and technical tutor. Provide clear, concise, and accurate answers.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--adapter-dir", type=str, default="/content/outputs/training/dataset-v2.0/qlora-v2/production/checkpoints/best")
    parser.add_argument("--no-adapter", action="store_true", help="Disable LoRA adapter and use base model")
    args, _ = parser.parse_known_args()

    # If /content/infer_request.json exists, override parameters
    req_file = Path("/content/infer_request.json")
    use_no_adapter = args.no_adapter

    if req_file.exists():
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                req_data = json.load(f)
            prompt = req_data.get("prompt", args.prompt)
            system_prompt = req_data.get("system_prompt", args.system_prompt)
            max_tokens = int(req_data.get("max_tokens", args.max_tokens))
            temperature = float(req_data.get("temperature", args.temperature))
            top_p = float(req_data.get("top_p", args.top_p))
            model_id = req_data.get("model_id", args.model_id)
            adapter_dir = req_data.get("adapter_dir", args.adapter_dir)
            use_no_adapter = bool(req_data.get("no_adapter", args.no_adapter))
        except Exception:
            prompt = args.prompt
            system_prompt = args.system_prompt
            max_tokens = args.max_tokens
            temperature = args.temperature
            top_p = args.top_p
            model_id = args.model_id
            adapter_dir = args.adapter_dir
    else:
        prompt = args.prompt
        system_prompt = args.system_prompt
        max_tokens = args.max_tokens
        temperature = args.temperature
        top_p = args.top_p
        model_id = args.model_id
        adapter_dir = args.adapter_dir

    if not prompt:
        print("❌ Error: No prompt provided.")
        return 1

    print("⏳ Initializing PyTorch CUDA runtime environment on Tesla T4 GPU...", flush=True)
    import torch
    print("⏳ Loading Transformers & Peft libraries...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Tokenizer
    print(f"⏳ Loading Tokenizer for '{model_id}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fast FP16 Base Model (6GB VRAM footprint on T4 GPU)
    print(f"⏳ Loading Base Model '{model_id}' into VRAM (device: {device})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to("cpu")
    print("✓ Base Model weights loaded successfully into VRAM.", flush=True)

    adapter_attached = False
    if not use_no_adapter:
        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            fallback = PROJECT_ROOT / "outputs/training/dataset-v2.0/qlora-v2/production/checkpoints/best"
            if fallback.exists():
                adapter_path = fallback

        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
            adapter_attached = True

    model.eval()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Configure explicit ChatML EOS and Stop Tokens
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for token_str in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and isinstance(tid, int) and tid not in eos_ids and tid >= 0:
                eos_ids.append(tid)
        except Exception:
            pass

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.7,
            top_p=top_p if temperature > 0 else 0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = out[0][input_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Clean special stop strings
    for stop_str in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]:
        if stop_str in answer:
            answer = answer.split(stop_str)[0].strip()

    # Truncate if model simulates subsequent turns
    for marker in [
        "\nuser", "\nUser:", "\nHuman:", "\n### User", "\n### Human",
        "\nassistant", "\nAssistant:", "\n### Assistant", "\nSystem:",
        "user\n", "assistant\n"
    ]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    answer = re.sub(r'(\s*----\s*){2,}', '', answer).strip()
    answer = re.sub(r'(\s*====\s*){2,}', '', answer).strip()

    # Save to disk for guaranteed recovery on connection drops
    res_path = Path("/content/infer_response.txt")
    try:
        res_path.write_text(answer, encoding="utf-8")
    except Exception:
        pass

    mode_label = "LoRA Fine-Tuned Adapter" if adapter_attached else "Base Model Only"
    print("\n" + "=" * 60, flush=True)
    print(f"🤖 QWEN3-4B RESPONSE [{mode_label}] (Tesla T4 GPU):", flush=True)
    print("=" * 60, flush=True)
    print(answer, flush=True)
    print("=" * 60, flush=True)
    print(f"⏱️ Inference Latency: {elapsed:.2f}s | Device: {device.upper()}", flush=True)


if __name__ == "__main__":
    main()
