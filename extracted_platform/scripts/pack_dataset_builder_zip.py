#!/usr/bin/env python3
"""
Packs the PDF/HTML/MD Document-to-Dataset Pipeline into a standalone ZIP package for Google Colab.
Output: dataset_builder_colab.zip
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INCLUDE_PATTERNS = [
    "configs/*.yaml",
    "configs/**/*.yaml",
    "src/**/*.py",
    "scripts/process_documents_to_dataset.py",
]

EXCLUDE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.git/**",
    "**/*.zip",
]

REQUIREMENTS_CONTENT = """# Dependencies for Document-to-Dataset Pipeline on Google Colab
pypdf>=3.10.0
pymupdf>=1.22.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
pyyaml>=6.0
pydantic>=2.0.0
tqdm>=4.65.0
markdown>=3.4.0
rouge-score>=0.1.2
"""

README_COLAB_CONTENT = """# 📄 PDF / HTML / Markdown to LLM Fine-Tuning Dataset Pack (Colab Guide)

This package converts raw **PDF**, **HTML**, **Markdown (`.md`)**, **Text (`.txt`)**, and **JSON** documents into a final, production-ready instruction dataset (`train.jsonl`, `validation.jsonl`, `test.jsonl`) ready for QLoRA fine-tuning in Google Colab.

---

## 🚀 How to Run on Google Colab (Step-by-Step)

### Step 1: Upload `dataset_builder_colab.zip` to Colab
In your Colab Notebook cell:
```python
# 1. Unzip the package
!unzip -q dataset_builder_colab.zip -d /content/dataset_builder

# 2. Change directory
%cd /content/dataset_builder

# 3. Install required dependencies
!pip install -r requirements.txt
```

---

### Step 2: Upload your Raw Documents (PDF / HTML / MD / TXT)
Create an input folder and place your documents into it:
```bash
!mkdir -p /content/my_raw_documents
```
*Upload your PDF, HTML, or `.md` files to `/content/my_raw_documents/` via the Colab File Explorer or Google Drive.*

---

### Step 3: Run the Master Dataset Builder Script
Run the single command below to process all your documents into a final training dataset:

```bash
!python process_documents_to_dataset.py \
    --input /content/my_raw_documents \
    --output-dir /content/final_training_dataset \
    --source my_custom_corpus \
    --seed 42
```

---

## 📦 What files will be generated?

Inside `/content/final_training_dataset/`:
1. **`train.jsonl`** — 90% of your records formatted as ChatML / Q&A instruction pairs.
2. **`validation.jsonl`** — 5% held-out validation set.
3. **`test.jsonl`** — 5% test evaluation set.
4. **`dataset_manifest.json`** — Complete statistics (document counts, chunk count, train/val/test breakdown).

You can now use `train.jsonl` directly in your QLoRA / SFT training scripts on Google Colab!
"""


def pack_zip() -> Path:
    zip_path = PROJECT_ROOT / "dataset_builder_colab.zip"
    
    files_packed = 0
    seen_paths = set()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Standard python files & configs
        for pattern in INCLUDE_PATTERNS:
            for file_path in PROJECT_ROOT.glob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    
                    # Store process_documents_to_dataset.py at root of zip
                    if rel_path == "scripts/process_documents_to_dataset.py":
                        arc_name = "process_documents_to_dataset.py"
                    else:
                        arc_name = rel_path

                    if arc_name in seen_paths:
                        continue
                    if any(file_path.match(ex) for ex in EXCLUDE_PATTERNS):
                        continue

                    zipf.write(file_path, arcname=arc_name)
                    seen_paths.add(arc_name)
                    files_packed += 1

        # 2. Add requirements.txt
        zipf.writestr("requirements.txt", REQUIREMENTS_CONTENT)
        files_packed += 1

        # 3. Add README_COLAB.md
        zipf.writestr("README_COLAB.md", README_COLAB_CONTENT)
        files_packed += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ Successfully created ZIP archive: {zip_path} ({size_mb:.2f} MB, {files_packed} items packed)")
    return zip_path


if __name__ == "__main__":
    pack_zip()
