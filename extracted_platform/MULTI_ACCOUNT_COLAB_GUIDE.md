# Multi-Account Stateless GPU Setup for Google Colab Free Tier

This architecture decouples **Compute (Hardware)** from **Storage (Model & Checkpoints)** so you can switch Google accounts freely without losing training progress.

---

## Architecture Overview

```mermaid
graph TD
    subgraph Storage [Primary Google Drive (Account A)]
        M[Base Model Weights]
        D[FROZEN dataset-v2.0]
        C[Checkpoints: checkpoint-25, checkpoint-50...]
    end

    subgraph Compute [Stateless Colab GPU Workers]
        W1[Colab Account 1 (T4 GPU)]
        W2[Colab Account 2 (T4 GPU)]
        W3[Colab Account 3 (T4 GPU)]
    end

    W1 -.->|1. Mount & Train| Storage
    W1 -.->|Quota Limit Hit| W2
    W2 -.->|2. Auto-Resume from latest checkpoint| Storage
    W2 -.->|Quota Limit Hit| W3
    W3 -.->|3. Complete Training & Save Best Model| Storage
```

---

## 2-Minute Setup for Multiple Accounts

### Step 1: Share Drive Folder from Primary Account
1. Open [Google Drive](https://drive.google.com) on your **Primary Account** (where your model and dataset are stored).
2. Right-click the **`GoogleColab`** folder $\rightarrow$ **Share**.
3. Add your other Google accounts (Account B, Account C, etc.) with **Editor** permissions.

### Step 2: Add Shortcut in Secondary Accounts
1. Log into Google Drive with **Account B** (or C).
2. Click **"Shared with me"** on the left menu.
3. Right-click the shared **`GoogleColab`** folder $\rightarrow$ Click **"Add shortcut to Drive"** $\rightarrow$ Select **"My Drive"**.

*(Now, `drive.mount('/content/drive')` in ANY Colab account will see `/content/drive/MyDrive/GoogleColab/AI/Qwen3` identically).*

---

## One-Cell Colab Execution (Run in ANY Account)

Open a new Colab notebook in any Google account, set runtime to **T4 GPU**, and run:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Check GPU & Project
import os, sys, subprocess
from pathlib import Path

# Verify Drive Path
qwen_dir = Path("/content/drive/MyDrive/GoogleColab/AI/Qwen3")
print("Google Drive Connected:", qwen_dir.exists())

# 3. Launch / Auto-Resume Training
# (train_production_v2.py automatically finds the latest checkpoint and resumes)
!python /content/GoogleColab/scripts/train_production_v2.py --config /content/GoogleColab/configs/training_v2.yaml
```

---

## Automated Multi-Account Token Vault & GPU Pool Failover Manager

We have created an automated script: `scripts/colab_account_manager.py`.

### How the Token Vault Works:
1. **Permanent Token Backup (`~/.config/colab-cli/saved_accounts/`)**: Every Google Account you log into is automatically backed up into your local Vault. You never need to re-authenticate an account twice.
2. **Automatic GPU Pool Failover**: When allocating a Tesla T4 GPU, the script tests all saved tokens in your Vault sequentially. If Account 1 is rate-limited (503/412), it instantly switches to Account 2, Account 3, etc.
3. **Interactive Login for New Accounts**: If all Vault tokens are rate-limited, it prompts for a fresh Google Account login, which is automatically saved to the Vault for future use.

### Run Commands:
* **Automatic Failover & Training Launch**:
  ```bash
  python scripts/colab_account_manager.py
  ```
* **List All Saved Google Accounts in Vault**:
  ```bash
  python scripts/colab_account_manager.py --list-vault
  ```
* **Connect GPU Pool Without Auto-Starting Training**:
  ```bash
  python scripts/colab_account_manager.py --no-train
  ```
