# 🚀 Complete End-to-End LLM Pipeline Guide
## From Raw Documents (PDF/HTML/TXT/JSONL) to Production QLoRA Fine-Tuning & Inference

This comprehensive guide outlines the complete end-to-end technical pipeline used in this project to transform raw unstructured documents (PDFs, HTML documentation, Markdown, Textbooks, Custom Q&A Datasets) into a production-grade instruction-tuned LLM (**Qwen3-4B / Qwen2.5-3B**) using 4-bit QLoRA on NVIDIA Tesla T4 GPUs.

---

## 🏛️ Pipeline Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Raw Data Ingestion (PDF / HTML / Text / Custom JSONL)│
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Parsing, Structure Extraction & Semantic Chunking    │
  │    - 512-1024 token chunks with context preservation     │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. Synthetic QA & Instruction Generation (ChatML)       │
  │    - 4 Difficulty Tiers: Beginner, Inter, Adv, Expert   │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. 10-Dimension Quality Audit & Leakage Guard            │
  │    - Domain & technical accuracy verification            │
  │    - Cross-split n-gram leakage prevention               │
  │    - Cryptographic SHA-256 Lock (FROZEN Lifecycle)       │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 5. Workspace Packaging & Remote Colab Sync              │
  │    - Compressed payload transfer to Tesla T4 GPU VM      │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 6. Production QLoRA Training Loop (Tesla T4 GPU)        │
  │    - 4-bit NF4 double quantization (~2.2 GB VRAM)        │
  │    - LoRA (r=16, alpha=32) on 7 linear projections       │
  │    - Assistant-only loss masking + PagedAdamW8bit        │
  └───────────────────────────┬─────────────────────────────┘
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 7. Checkpoints Download & Interactive Chat Inference     │
  │    - Best adapter weights (adapter_model.safetensors)    │
  └───────────────────────────┘
```

---

## 📋 Step-by-Step Command Execution Flow

### ⚡ Option A: Single Master Command (Automated Document-to-Dataset)
If you want to transform raw PDF, HTML, Markdown (`.md`), Text (`.txt`), or JSON documents into a final training dataset (`train.jsonl`, `validation.jsonl`, `test.jsonl`) in a **single command**, run:

```bash
python scripts/process_documents_to_dataset.py \
    --input data/raw_documents/ \
    --output-dir data/instruction_dataset/v3.0 \
    --source my_corpus \
    --seed 42
```
*This single script automatically combines Ingestion (Step 1) + Generation (Step 2) + Quality Audit & Splitting (Step 3).*

---

### 🧩 Option B: Modular Step-by-Step Execution

#### 📄 Step 1: Raw Document Ingestion & Semantic Chunking
Extracts clean text, formulas, and code from raw PDF/HTML/TXT documents and splits them into semantically coherent chunks:
```bash
python scripts/ingest_documents.py \
    --input data/raw_documents/ \
    --output data/instruction_dataset/v2.0/raw/candidates.jsonl \
    --chunk-size 1024
```

---

#### 🧠 Step 2: Synthetic QA Instruction Generation
Converts raw chunks into structured multi-turn / single-turn ChatML instruction pairs across 4 difficulty tiers:
```bash
python scripts/generate_instruction_dataset.py \
    --input data/ingested/v3.0/chunks.jsonl \
    --documents data/ingested/v3.0/documents.jsonl \
    --output-dir data/instruction_dataset/v3.0 \
    --seed 42
```

---

#### 🛡️ Step 3: Quality Audit, Leakage Filter & Dataset Freeze
Executes the **10-Dimension Release QA Gate Engine**:
* Domain & technical accuracy validation
* License compliance & attribution tracking
* Cross-split n-gram leakage prevention (Train / Validation / Test)
* Enforces `>= 0.85` quality threshold
* Locks dataset into **`🔒 FROZEN`** state with cryptographic SHA-256 checksums:

```bash
python scripts/build_dataset_v2.py \
    --input data/instruction_dataset/v3.0 \
    --version dataset-v3.0 \
    --output-dir data/instruction_dataset/v3.0 \
    --target 10000 \
    --freeze
```
> **Output Splitting**: Creates `train.jsonl` (90%), `validation.jsonl` (5%), and `test.jsonl` (5%).

---

### 📦 Step 4: Workspace Packaging
Packages the frozen dataset, configs, and training engine into a compressed ZIP payload (`data/workspace_sync.zip`):
```bash
python scripts/pack_sync_payload.py
```

---

### 🚀 Step 5: Multi-Account Colab GPU Allocation & QLoRA Training

#### Option A: Automatic Multi-Account Vault Manager (Recommended)
Automatically cycles through all saved Google accounts in your local Vault (`~/.config/colab-cli/saved_accounts/`). If an account hits GPU 503/412 rate-limits, it seamlessly fails over to the next saved account without re-logins:
```bash
python scripts/colab_account_manager.py
```
*List all saved accounts in your Vault:*
```bash
python scripts/colab_account_manager.py --list-vault
```

#### Option B: Direct Autonomous Colab Runner
Allocates remote Tesla T4 GPU session, auto-uploads payload via native chunked HTTP (`colab upload`), verifies hardware preflight, and executes 3-Epoch QLoRA training:
```bash
python scripts/run_colab_job.py --action train
```

---

### 📊 Step 6: Real-Time Live Telemetry Monitoring
Tracks live training progress, loss curve, epoch, learning rate, and GPU VRAM without opening Colab web UI:
```bash
python scripts/monitor_training.py
```

---

### 📥 Step 7: Checkpoints & Reports Sync
Downloads best LoRA adapter weights (`adapter_model.safetensors`), manifests, and evaluation summaries to your local machine:
```bash
python scripts/run_colab_job.py --action sync
```

---

### 💬 Step 8: Interactive Chat Inference
Test your fine-tuned model directly from the command line:
```bash
python scripts/chat_inference.py
```

---

## ⚙️ Core Technical Specifications

| Parameter | Configuration | Technical Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-3B` | High architectural compatibility with Qwen3-4B instruction pipeline |
| **Quantization** | `4-bit NF4` (Double Quant) | Reduces VRAM footprint from ~8 GB to **~2.2 GB VRAM** on Tesla T4 |
| **LoRA Config** | `r=16, alpha=32, dropout=0.05` | Trains **1.73%** parameters (29.9M params), preserving base knowledge |
| **Target Modules** | All 7 Projections | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Loss Masking** | Assistant-Only | Zero loss on user prompts; gradients computed strictly on output tokens |
| **Optimizer** | `PagedAdamW8bit` | Prevents CUDA out-of-memory spikes during peak gradient allocation |
| **Schedule** | Cosine with Warmup | LR: `2e-4`, Warmup: 23 steps, Total: 789 optimizer steps (3 Epochs) |
| **Validation** | Step-based (Every 25 steps) | Evaluates on 117 held-out validation records and tracks best loss |

---

## 🛠️ Quick Command Reference

| Action | Command |
| :--- | :--- |
| **Extract & Ingest** | `python scripts/ingest_documents.py --input <data_dir>` |
| **Generate QA** | `python scripts/generate_dataset.py --count 100 --difficulty intermediate` |
| **Validate & Freeze** | `python scripts/build_dataset_v2.py --input <candidates.jsonl> --freeze` |
| **Pack Workspace** | `python scripts/pack_sync_payload.py` |
| **Launch Training** | `python scripts/colab_account_manager.py` |
| **Live Monitor** | `python scripts/monitor_training.py` |
| **Pull Artifacts** | `python scripts/run_colab_job.py --action sync` |
| **Chat Inference** | `python scripts/chat_inference.py` |
