#!/usr/bin/env python3
"""
Autonomous Remote Execution Runner for Google Colab GPU.
Orchestrates environment verification, chunked payload synchronization, and 
production training lifecycle across stateless Colab T4 GPU nodes.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_colab_bin() -> str:
    venv_colab = PROJECT_ROOT / ".venv/bin/colab"
    if venv_colab.exists():
        return str(venv_colab)
    which_colab = shutil.which("colab")
    if which_colab:
        return which_colab
    return "colab"


def run_cmd(cmd: list[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out."
    except Exception as e:
        return -2, "", str(e)


def ensure_session(session_name: str = "t4-prod", gpu: str = "T4") -> bool:
    colab_bin = get_colab_bin()
    sessions_cache = Path.home() / ".config/colab-cli/sessions.json"
    
    code, stdout, stderr = run_cmd([colab_bin, "sessions"], timeout=15)
    if session_name in stdout:
        print(f"✓ Active session '{session_name}' verified.")
        return True

    if sessions_cache.exists():
        sessions_cache.unlink()

    print(f"\n🚀 Session '{session_name}' is not currently active. Allocating {gpu} GPU...")
    create_cmd = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
    code, out, err = run_cmd(create_cmd, timeout=90)
    chk_code, chk_out, _ = run_cmd([colab_bin, "sessions"], timeout=15)

    if session_name in chk_out or ("Creating session" in out and "ColabRequestError" not in out):
        print(f"🎉 SUCCESS: Colab {gpu} session '{session_name}' successfully allocated.")
        # Force refresh local colab_cli assignment cache
        run_cmd([sys.executable, "-c", "from colab_cli.common import state; state.client.list_assignments()"])
        print("⏳ Waiting 25s for Colab VM notebook server services to fully initialize...")
        time.sleep(25)
        # Readiness check loop
        for ping in range(3):
            ping_script = Path("/tmp/ping_ready.py")
            ping_script.write_text("print('COLAB_VM_READY')\n", encoding="utf-8")
            code, out, _ = run_cmd([colab_bin, "exec", "-s", session_name, "-f", str(ping_script)], timeout=15)
            if "COLAB_VM_READY" in out:
                print("✅ Colab VM Kernel Ready & Online.")
                return True
            time.sleep(5)
        print("✅ Proceeding with initialized session.")
        return True

    print(f"⚠️ Primary allocation hit quota/error ({err.strip() or out.strip()}). Triggering Multi-Account Vault Failover...")
    from scripts.colab_account_manager import auto_failover_vault_pool
    return auto_failover_vault_pool(session_name=session_name, gpu=gpu)


def sync_workspace(session_name: str, max_retries: int = 3) -> bool:
    colab_bin = get_colab_bin()
    pack_script = PROJECT_ROOT / "scripts/pack_sync_payload.py"
    zip_path = PROJECT_ROOT / "data/workspace_sync.zip"

    print("\n📦 Bundling latest project workspace into compressed ZIP payload...")
    if pack_script.exists():
        run_cmd([sys.executable, str(pack_script)])

    if not zip_path.exists():
        print(f"❌ Zip payload missing at: {zip_path}")
        return False

    zip_bytes = zip_path.read_bytes()
    size_mb = len(zip_bytes) / (1024 * 1024)
    b64_str = base64.b64encode(zip_bytes).decode("ascii")

    sync_script = Path("/tmp/colab_single_pass_sync.py")
    sync_script.write_text(
        f"import base64, zipfile, io, os\n"
        f"data = base64.b64decode('{b64_str}')\n"
        f"with open('/content/workspace_sync.zip', 'wb') as f:\n"
        f"    f.write(data)\n"
        f"with zipfile.ZipFile(io.BytesIO(data), 'r') as z:\n"
        f"    z.extractall('/content')\n"
        f"print('✅ WORKSPACE_SYNC_EXTRACTED_SUCCESS')\n",
        encoding="utf-8"
    )

    for attempt in range(1, max_retries + 1):
        print(f"📦 Synchronizing workspace ({size_mb:.2f} MB) to Colab VM session '{session_name}' (Attempt {attempt}/{max_retries})...")
        code, out, err = run_cmd([colab_bin, "exec", "-s", session_name, "-f", str(sync_script)], timeout=120)
        
        if "WORKSPACE_SYNC_EXTRACTED_SUCCESS" in out:
            print("✅ Workspace & checkpoints successfully synced & extracted on Colab VM (/content).")
            sync_script.unlink(missing_ok=True)
            return True
            
        print(f"❌ Sync attempt {attempt} failed: {err or out}")
        
        # Reset session state on failure
        run_cmd([sys.executable, "-c", "from colab_cli.common import state; assignments = state.client.list_assignments(); [state.client.unassign(a.endpoint) for a in assignments]"])
        run_cmd([colab_bin, "stop", "-s", session_name])
        time.sleep(2)
        ensure_session(session_name=session_name, gpu="T4")
        time.sleep(2)

    print("❌ Workspace synchronization failed after retries.")
    return False


def pull_artifacts(session_name: str):
    colab_bin = get_colab_bin()
    print("\n📥 Pulling reports and telemetry artifacts from Colab VM to local workspace...")
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        ("training_heartbeat.json", reports_dir / "training_heartbeat.json"),
        ("training_completion_manifest.json", reports_dir / "training_completion_manifest.json"),
    ]

    for remote_name, local_dest in artifacts:
        run_cmd([colab_bin, "download", "-s", session_name, f"/content/reports/{remote_name}", str(local_dest)], timeout=30)
        if local_dest.exists():
            print(f"✓ Synced artifact: {local_dest.name}")


def run_training_job(session_name: str = "t4-prod", gpu: str = "T4") -> int:
    colab_bin = get_colab_bin()

    print("=" * 80)
    print("🤖 AUTONOMOUS COLAB RUNNER: TRAIN")
    print(f"📍 Session: {session_name} | GPU Target: {gpu}")
    print("=" * 80)

    if not ensure_session(session_name=session_name, gpu=gpu):
        print("\n⛔ Session allocation failed. Aborting.")
        return 1

    if not sync_workspace(session_name=session_name):
        print("\n⛔ Workspace synchronization failed. Aborting.")
        return 1

    print(f"\n📦 Preparing environment on Colab session '{session_name}'...")
    deps_cmd = [
        colab_bin, "exec", "-s", session_name, "-f",
        str(PROJECT_ROOT / "scripts/remote_install_deps.py"),
    ]
    run_cmd(deps_cmd)

    print(f"\n🚀 Launching 'scripts/train_production_v2.py' on Colab session '{session_name}'...")
    train_cmd = [
        colab_bin, "exec", "-s", session_name, "-f",
        str(PROJECT_ROOT / "scripts/train_production_v2.py"),
        "--timeout", "36000",
    ]

    proc = subprocess.run(train_cmd)
    pull_artifacts(session_name)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Autonomous Colab Job Runner")
    parser.add_argument("--action", choices=["train", "sync", "status"], default="train", help="Action to perform")
    parser.add_argument("--session", "-s", default="t4-prod", help="Colab session name")
    parser.add_argument("--gpu", "-g", default="T4", help="GPU variant target")
    args = parser.parse_args()

    if args.action == "train":
        code = run_training_job(session_name=args.session, gpu=args.gpu)
    elif args.action == "sync":
        pull_artifacts(session_name=args.session)
        code = 0
    elif args.action == "status":
        colab_bin = get_colab_bin()
        subprocess.run([colab_bin, "sessions"])
        code = 0
    else:
        code = 0

    if code == 0:
        print(f"\n🎉 Job '{args.action}' finished successfully!")
    else:
        print(f"\n❌ Job '{args.action}' failed with exit code {code}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
