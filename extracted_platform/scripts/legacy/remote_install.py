import subprocess
import sys

print("Installing bitsandbytes and checking dependencies...")
try:
    res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], capture_output=True, text=True)
    print("pip install stdout:", res.stdout)
    if res.stderr:
        print("pip install stderr:", res.stderr)
    print("pip return code:", res.returncode)
except Exception as e:
    print("Failed to run pip:", e)

import bitsandbytes as bnb
print("bitsandbytes successfully imported! Version:", getattr(bnb, "__version__", "unknown"))
