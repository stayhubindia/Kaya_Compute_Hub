import sys
import subprocess

print("📦 Auto-installing Colab production dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1", "peft>=0.14.0", "transformers>=4.49.0", "accelerate", "datasets", "pyyaml", "trl"])
print("✓ Colab dependencies ready.")
