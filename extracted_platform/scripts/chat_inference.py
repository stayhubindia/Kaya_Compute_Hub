#!/usr/bin/env python3
"""
Interactive Chat & Prompt Inference Utility for Qwen3-4B-Base (Phase 5.1).
Supports prompt evaluation, comparative generation (Base vs LoRA), and interactive terminal chat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable Colab secret vault warnings
os.environ["HF_HUB_DISABLE_COLAB_SECRET_ACCESS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_hf_token() -> Optional[str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-4B LoRA Interactive Inference & Chat Engine")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Base model Hugging Face ID or local path",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="outputs/training/dataset-v2.0/qlora-v2/production/checkpoints/best",
        help="Path to trained LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Single prompt to evaluate (if not provided, launches interactive chat session)",
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        default="You are an expert AI assistant and domain specialist. Provide clear, concise, and accurate answers.",
        help="System instruction prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per response (default: 512)",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run base model only without attaching LoRA adapter",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate responses from both Base Model and LoRA Adapter side-by-side",
    )
    parser.add_argument(
        "--colab",
        "--remote",
        action="store_true",
        default=True,
        help="Execute inference on active Colab Tesla T4 GPU (zero local model download)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force running inference locally on local CPU/GPU (downloads base model if not cached)",
    )
    parser.add_argument(
        "--session",
        "-se",
        type=str,
        default="t4-prod",
        help="Colab session name (default: t4-prod)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device target (for local execution)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, adapter_dir: Optional[Path], use_adapter: bool, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = get_hf_token()
    print(f"📦 Loading Tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available() and device != "cpu"
    compute_device = "cuda" if has_cuda else "cpu"
    torch_dtype = torch.float16 if has_cuda else torch.float32

    print(f"📦 Loading Base Model ({compute_device.upper()} / {torch_dtype})...")
    if has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=hf_token,
            trust_remote_code=True,
        )
        model.to("cpu")

    if use_adapter and adapter_dir and adapter_dir.exists():
        print(f"✨ Attaching LoRA Adapter from: {adapter_dir}...")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print("✓ LoRA Adapter attached successfully.")

    model.eval()
    return model, tokenizer, compute_device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if device == "cuda" or (torch.cuda.is_available() and hasattr(model, "device")):
        inputs = {k: v.to(model.device if hasattr(model, "device") else "cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated response
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def main() -> int:
    args = parse_args()
    adapter_path = Path(args.adapter_dir)
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path

    print("=" * 70)
    print("🚀 QWEN3-4B INTERACTIVE INFERENCE & CHAT CONSOLE")
    print("=" * 70)
    print(f"Target Base Model: {args.model_id}")
    print(f"LoRA Adapter Dir:  {adapter_path} (Exists: {adapter_path.exists()})")
    
    # Check if running remotely on Colab T4 GPU (zero local model download)
    if not args.local:
        from scripts.run_colab_job import ensure_session, sync_workspace, get_colab_bin, run_cmd
        colab_bin = get_colab_bin()

        import subprocess
        import tempfile
        print(f"🌐 Mode: Colab T4 GPU Remote Inference (Session: {args.session})")
        print("💡 (Zero local download required — running on Colab Tesla T4 NVMe SSD)")
        print("=" * 70)

        remote_script_path = PROJECT_ROOT / "scripts/remote_infer.py"

        def run_remote_inference(prompt_text: str):
            ensure_session(session_name=args.session, gpu="T4")

            # Check if remote_infer.py exists on Colab VM, if not sync workspace
            code, out, _ = run_cmd([colab_bin, "exec", "-s", args.session, "-f", str(PROJECT_ROOT / "scripts/list_remote_checkpoints.py")])
            if code != 0 or "Checking directory" not in out:
                sync_workspace(session_name=args.session)

            req_data = {
                "prompt": prompt_text,
                "system_prompt": args.system_prompt,
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "model_id": args.model_id,
                "adapter_dir": "/content/outputs/training/dataset-v2.0/qlora-v2/production/checkpoints/best",
                "no_adapter": args.no_adapter,
            }
            tmp_file = PROJECT_ROOT / "infer_request.json"
            with open(tmp_file, "w", encoding="utf-8") as tmp:
                json.dump(req_data, tmp, indent=2)

            # Upload request payload to Colab root
            subprocess.run([str(colab_bin), "upload", "-s", args.session, str(tmp_file), "infer_request.json"], capture_output=True)

            # Execute remote_infer.py on Colab GPU
            cmd = [str(colab_bin), "exec", "-s", args.session, "-f", str(remote_script_path), "--timeout", "120"]
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                # Check if infer_response.txt was written on remote VM
                r_code, r_out, _ = run_cmd([colab_bin, "exec", "-s", args.session, "-c", "import os; print(open('/content/infer_response.txt').read() if os.path.exists('/content/infer_response.txt') else 'NO_RESPONSE')"])
                if r_code == 0 and "NO_RESPONSE" not in r_out and len(r_out.strip()) > 5:
                    print("\n" + "=" * 60)
                    print(f"🤖 QWEN3-4B RESPONSE (Tesla T4 GPU):")
                    print("=" * 60)
                    print(r_out.strip())
                    print("=" * 60)

        if args.prompt:
            print(f"\n👤 USER PROMPT:\n{args.prompt}")
            print("\n🤖 GENERATING RESPONSE VIA COLAB T4 GPU...")
            run_remote_inference(args.prompt)
            return 0

        # Interactive loop via Colab
        print("\n💬 Entering Remote Colab Chat Mode. Type 'exit', 'quit', or press Ctrl+C to stop.\n")
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("👋 Exiting chat session.")
                    break

                run_remote_inference(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Session ended.")
                break
        return 0

    print(f"💻 Mode: Local Machine Inference (Device: {args.device})")
    print("=" * 70)

    use_adapter = not args.no_adapter and adapter_path.exists()
    model, tokenizer, device = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_dir=adapter_path if use_adapter else None,
        use_adapter=use_adapter,
        device=args.device,
    )

    # Single Prompt Mode
    if args.prompt:
        print("\n" + "-" * 70)
        print(f"👤 USER PROMPT:\n{args.prompt}")
        print("-" * 70)
        print("🤖 GENERATING RESPONSE...")
        t0 = time.perf_counter()
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n💡 ASSISTANT:\n{answer}")
        print("-" * 70)
        print(f"⏱️ Generated in {elapsed:.2f}s")
        return 0

    # Interactive Chat Loop
    print("\n💬 Entering Interactive Chat Mode. Type 'exit', 'quit', or press Ctrl+C to stop.\n")
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting chat session.")
                break

            print("\n🤖 Qwen Assistant:")
            t0 = time.perf_counter()
            answer = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            elapsed = time.perf_counter() - t0
            print(answer)
            print(f"\n[⏱️ {elapsed:.2f}s]")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
