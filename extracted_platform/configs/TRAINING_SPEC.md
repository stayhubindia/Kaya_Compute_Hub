# Qwen3-4B-Base QLoRA Supervised Fine-Tuning Specification (Phase 4.1)

## 1. Executive Summary & Objective

This document defines the production fine-tuning specification, dataset validation architecture, tokenization format, parameter accounting, and preflight readiness protocol for **Qwen3-4B-Base** using **4-bit QLoRA (NF4)**.

All dataset ingestion is strictly governed by the cryptographically frozen production dataset `dataset-v1.0` verified through SHA-256 signatures. Training execution is prohibited unless all preflight gates pass.

---

## 2. Hardware Environment & Feasibility Envelope

Fine-tuning is designed for single-GPU execution on an **NVIDIA Tesla T4 (16 GB GDDR6 VRAM)** in Google Colab, with automated fallback execution and offline mock verification on CPU/dev environments.

### 2.1 VRAM Allocation Budget (Tesla T4 16GB)

| Component | Precision / Configuration | VRAM Footprint | Description |
| :--- | :--- | :--- | :--- |
| **Base Model Weights** | 4-bit NormalFloat (NF4) + Double Quant | **~2.45 GB** | Qwen3-4B base parameters quantized to 4-bit |
| **LoRA Adapters ($r=16$)** | FP16 / BF16 ($25.95\text{M}$ params) | **~0.10 GB** | Trainable projection adapters |
| **Optimizer States (AdamW)**| 8-bit Paged AdamW / FP32 | **~0.21 GB** | First & second moments for adapter weights |
| **Activation Memory** | Micro-batch size $1$, seq len $4096$, grad ckpt | **~2.80 GB** | Checkpointed intermediate activations |
| **CUDA Kernels & Workspace**| PyTorch context & cache | **~0.60 GB** | CUDA runtime overhead |
| **Total Estimated Peak VRAM**| — | **~6.16 GB** | **Well within 16.0 GB budget ($< 40\%$ utilization)** |

---

## 3. Dataset Ingestion & Integrity Enforcement

### 3.1 Immutability & Lifecycle Lock
The dataset loader (`TrainingDatasetLoader`) mandates that the target production dataset release must be in the `FROZEN` lifecycle state:
1. `production_manifest.json` status must equal `FROZEN`.
2. All split files (`train.jsonl`, `validation.jsonl`, `test.jsonl`) are verified against recorded SHA-256 signatures prior to loading.
3. Every record must conform strictly to `DatasetRecord` Pydantic schema with complete `ProvenanceInfo`.

### 3.2 Split Isolation & Leakage Prevention
Zero cross-split hash collisions are verified at load time across all sets:
$$\mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{val}} = \emptyset, \quad \mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{test}} = \emptyset, \quad \mathcal{H}_{\text{val}} \cap \mathcal{H}_{\text{test}} = \emptyset$$

---

## 4. Tokenization & Assistant-Only Loss Masking

### 4.1 Native ChatML Formatting
Conversations are serialized using the native Qwen ChatML template:

```text
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
What is binary search?<|im_end|>
<|im_start|>assistant
Binary search is a divide-and-conquer algorithm with O(log n) time complexity.<|im_end|>
```

### 4.2 Assistant-Only Loss Collator
The custom `DataCollatorForAssistantOnlyLoss` ensures only assistant tokens contribute to the SFT cross-entropy loss:
* **System Prompt & Special Tokens:** `label = -100` (ignored by cross-entropy).
* **User Query & Special Tokens:** `label = -100` (ignored by cross-entropy).
* **Assistant Turn Prefix (`<|im_start|>assistant\n`):** `label = -100`.
* **Assistant Response Tokens + `<|im_end|>`:** `label = input_ids[i]`.
* **Padding Tokens:** `label = -100`, `attention_mask = 0`.

---

## 5. QLoRA Quantization & Adapter Architecture

### 5.1 4-bit Quantization Config (BitsAndBytes)
* **Quantization Type:** NormalFloat4 (`nf4`)
* **Double Quantization:** Enabled (`bnb_4bit_use_double_quant = True`)
* **Compute Dtype:** `bfloat16` if Ampere+ / `float16` for Turing (Tesla T4)

### 5.2 LoRA Hyperparameters
* **Rank ($r$):** `16`
* **LoRA Alpha ($\alpha$):** `32` ($\alpha / r = 2.0$ scaling factor)
* **LoRA Dropout:** `0.05`
* **Target Modules:** All 7 projection matrices in self-attention and MLP:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

### 5.3 Trainable Parameter Accounting
| Metric | Value |
| :--- | :--- |
| **Total Base Parameters** | $4,000,000,000$ |
| **LoRA Trainable Parameters** | $25,952,256$ |
| **Trainable Percentage** | **$0.6446\%$** |

---

## 6. Training Hyperparameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `num_train_epochs` | `3` | Standard convergence for conversational SFT |
| `per_device_train_batch_size` | `1` | Minimizes peak activation VRAM on T4 |
| `gradient_accumulation_steps` | `8` | Yields effective batch size of $8$ |
| `learning_rate` | `2e-4` ($0.0002$) | Standard QLoRA learning rate for rank 16 |
| `lr_scheduler_type` | `cosine` | Smooth decay with warmup |
| `warmup_ratio` | `0.03` ($3\%$) | Stabilizes initial gradient steps |
| `weight_decay` | `0.01` | L2 regularization on adapter weights |
| `gradient_checkpointing` | `true` | Essential for sequence lengths up to 4096 on 16GB VRAM |
| `save_total_limit` | `3` | Retains top 3 checkpoints, prunes older ones |

---

## 7. Execution CLI & Verification Protocol

### 7.1 CLI Commands

```bash
# 1. Run full 16-point preflight readiness audit
python scripts/train_qwen.py --preflight

# 2. Run single-batch forward pass dry run (memory & loss profile)
python scripts/train_qwen.py --dry-run

# 3. Generate sealed training manifest
python scripts/train_qwen.py --generate-manifest

# 4. Full Phase 4.1 end-to-end verification
python scripts/train_qwen.py --preflight --dry-run --generate-manifest
```

### 7.2 Safety & Protection Guarantees
* **No Unintentional Training Runs:** Full training requires the explicit `--train` flag.
* **Manifest Locking:** Checkpoint manager enforces dataset version lock; resuming across differing datasets triggers an immediate validation exception.
* **Audit Trail:** Every preflight report and training manifest is saved as JSON in `outputs/` and `datasets/production/manifests/`.
