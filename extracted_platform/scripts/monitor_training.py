#!/usr/bin/env python3
"""
Live Real-Time Telemetry Monitor & Checkpoint Sync Daemon for Colab Training.
Polls /content/reports/training_heartbeat.json, displays a rich status bar,
and AUTOMATICALLY downloads remote checkpoints to local laptop SSD storage in real-time.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_COLAB = PROJECT_ROOT / ".venv/bin/colab"
COLAB_BIN = str(VENV_COLAB) if VENV_COLAB.exists() else "colab"

_cached_heartbeat = None
_last_synced_step = -1

def fetch_heartbeat(session: str = "t4-prod") -> dict | None:
    global _cached_heartbeat
    tmp_dest = Path("/tmp/colab_heartbeat_live.json")
    
    cmd = [COLAB_BIN, "download", "-s", session, "/content/reports/training_heartbeat.json", str(tmp_dest)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if res.returncode == 0 and tmp_dest.exists():
            data = json.loads(tmp_dest.read_text(encoding="utf-8"))
            _cached_heartbeat = data
            return data
    except Exception:
        pass
        
    return _cached_heartbeat


def auto_sync_checkpoints_to_local(session: str = "t4-prod", current_step: int = 0):
    """Auto-pull remote checkpoints from Colab VM to local laptop SSD."""
    global _last_synced_step
    if current_step <= 0 or current_step == _last_synced_step:
        return

    # Create remote zipping script
    remote_script = Path("/tmp/remote_zip_ckpts.py")
    remote_script.write_text(
        "import os, zipfile\n"
        "ckpt_dir = '/content/outputs/training/dataset-v2.0/qlora-v2/production/checkpoints'\n"
        "zip_out = '/content/remote_ckpts.zip'\n"
        "if os.path.exists(ckpt_dir):\n"
        "    with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
        "        for root, dirs, files in os.walk(ckpt_dir):\n"
        "            for f in files:\n"
        "                full_p = os.path.join(root, f)\n"
        "                rel_p = os.path.relpath(full_p, '/content')\n"
        "                zf.write(full_p, rel_p)\n"
        "    print('CKPT_ZIP_SUCCESS')\n"
        "else:\n"
        "    print('NO_CKPT_DIR')\n",
        encoding="utf-8"
    )

    try:
        exec_cmd = [COLAB_BIN, "exec", "-s", session, "-f", str(remote_script)]
        res = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=30)
        if "CKPT_ZIP_SUCCESS" in res.stdout:
            local_tmp_zip = Path("/tmp/local_remote_ckpts.zip")
            dl_cmd = [COLAB_BIN, "download", "-s", session, "/content/remote_ckpts.zip", str(local_tmp_zip)]
            dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)
            
            if dl_res.returncode == 0 and local_tmp_zip.exists():
                with zipfile.ZipFile(local_tmp_zip, "r") as zf:
                    zf.extractall(PROJECT_ROOT)
                _last_synced_step = current_step
                ts = time.strftime("%H:%M:%S")
                print(f"\n[{ts}] 💾 Auto-Synced Checkpoint (Step {current_step}) to Local Laptop SSD! (/media/durgesh/Development/GoogleColab/outputs/)")
    except Exception as e:
        pass


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "t4-prod"
    print("=" * 70)
    print(f"📊 LIVE TRAINING MONITOR & CHECKPOINT AUTO-SYNC DAEMON (Session: {session})")
    print("   Press Ctrl+C to exit monitor (training continues in background)")
    print("=" * 70)

    while True:
        hb = fetch_heartbeat(session)
        if hb:
            step = hb.get("step", 0)
            total_steps = hb.get("total_steps", 861)
            epoch = hb.get("epoch", 0.0)
            loss = hb.get("current_loss", 0.0)
            vram = hb.get("vram_allocated_mb", 0.0)
            state = hb.get("state", "UNKNOWN")
            lr = hb.get("learning_rate", 0.0)

            pct = (step / total_steps) * 100 if total_steps else 0
            bar_len = 25
            filled = int(bar_len * step / max(1, total_steps))
            bar = "█" * filled + "░" * (bar_len - filled)

            ts = time.strftime("%H:%M:%S")
            print(f"\r[{ts}] [{bar}] {pct:5.1f}% | Step {step:3d}/{total_steps} | Ep {epoch:.2f} | Loss: {loss:.4f} | VRAM: {vram:.0f}MB | LR: {lr:.1e} | State: {state}", end="", flush=True)

            # Auto-sync checkpoints to local laptop SSD whenever a save_step (every 25 steps) occurs
            if step > 0 and step % 25 == 0:
                auto_sync_checkpoints_to_local(session=session, current_step=step)

            if state in ["COMPLETED", "FAILED"]:
                print(f"\n🎉 Training finished with state: {state}")
                # Final sync of checkpoints and reports
                auto_sync_checkpoints_to_local(session=session, current_step=step)
                break
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"\r[{ts}] ⏳ Waiting for training process to write initial heartbeat...", end="", flush=True)

        time.sleep(4)


if __name__ == "__main__":
    main()
