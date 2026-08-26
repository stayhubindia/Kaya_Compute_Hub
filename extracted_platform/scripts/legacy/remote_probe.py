import sys
import subprocess
import torch

print("Python version:", sys.version)
try:
    smi = subprocess.check_output(["nvidia-smi"]).decode()
    print("NVIDIA-SMI OUTPUT:")
    print(smi)
except Exception as e:
    print("nvidia-smi failed:", e)

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Compute Capability:", torch.cuda.get_device_capability(0))
    print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
