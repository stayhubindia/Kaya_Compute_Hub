#!/usr/bin/env python3
"""
Colab Multi-Account & GPU Failover Vault Manager.
Automatically saves, manages, and cycles through stored Google account tokens
to bypass free GPU rate-limits seamlessly without requiring constant re-logins.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

COLAB_CONFIG_DIR = Path.home() / ".config/colab-cli"
TOKEN_FILE = COLAB_CONFIG_DIR / "token.json"
SESSIONS_FILE = COLAB_CONFIG_DIR / "sessions.json"
TOKEN_VAULT_DIR = COLAB_CONFIG_DIR / "saved_accounts"


def ensure_dirs():
    COLAB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_VAULT_DIR.mkdir(parents=True, exist_ok=True)


def get_colab_bin() -> str:
    venv_colab = Path(__file__).resolve().parent.parent / ".venv/bin/colab"
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


def save_active_token_to_vault(name: Optional[str] = None) -> Optional[Path]:
    """Backup active token to vault."""
    ensure_dirs()
    if TOKEN_FILE.exists() and TOKEN_FILE.stat().st_size > 50:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = name or f"account_{ts}"
        dest_path = TOKEN_VAULT_DIR / f"{label}.json"
        shutil.copy2(TOKEN_FILE, dest_path)
        print(f"💾 Saved Google Account token to Vault: {dest_path.name}")
        return dest_path
    return None


def get_vault_tokens() -> List[Path]:
    ensure_dirs()
    return sorted(TOKEN_VAULT_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)


def load_vault_token(token_path: Path) -> bool:
    """Load a specific saved token into active position."""
    ensure_dirs()
    if token_path.exists():
        shutil.copy2(token_path, TOKEN_FILE)
        if SESSIONS_FILE.exists():
            SESSIONS_FILE.unlink()
        print(f"🔑 Switched active authentication to Vault Account: [{token_path.stem}]")
        return True
    return False


def clear_active_colab_auth():
    ensure_dirs()
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    if SESSIONS_FILE.exists():
        SESSIONS_FILE.unlink()


def try_create_session(session_name: str = "t4-prod", gpu: str = "T4") -> Tuple[bool, str]:
    colab_bin = get_colab_bin()
    
    if not TOKEN_FILE.exists():
        return False, "No active token file."

    # Stop orphan local session mapping first
    run_cmd([colab_bin, "stop", "-s", session_name], timeout=15)
    time.sleep(1)

    cmd = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
    code, stdout, stderr = run_cmd(cmd, timeout=60)
    full_output = f"{stdout}\n{stderr}"

    if code == 0 and "Creating session" in full_output and "ColabRequestError" not in full_output:
        return True, "Session created successfully."

    if "503" in full_output or "Service Unavailable" in full_output:
        return False, "503 Service Unavailable (GPU pool rate-limited / quota exhausted)"
    if "412" in full_output or "TooManyAssignmentsError" in full_output:
        return False, "412 Quota limit reached on this Google account"
    if "401" in full_output or "403" in full_output or "unauthorized" in full_output.lower():
        return False, "Authentication token expired or unauthorized"

    return False, f"Allocation failed: {stderr.strip() or stdout.strip()}"


def auto_failover_vault_pool(session_name: str = "t4-prod", gpu: str = "T4") -> bool:
    """
    1. Tries active token.
    2. If failed, loops through all saved tokens in ~/.config/colab-cli/saved_accounts/
    3. Returns True if any account successfully allocates GPU.
    """
    ensure_dirs()
    
    # 1. Save whatever is currently active first
    save_active_token_to_vault()

    vault_tokens = get_vault_tokens()
    print(f"\n🔐 Vault contains {len(vault_tokens)} saved Google Account tokens.")

    # Try existing active token first
    print("\n[1/Pool] Testing currently active Google Account...")
    success, reason = try_create_session(session_name=session_name, gpu=gpu)
    if success:
        print(f"🎉 SUCCESS! GPU Allocated on active account.")
        return True
    print(f"⚠️ Active account unavailable: {reason}")

    # Loop through vault
    for idx, t_path in enumerate(vault_tokens, start=2):
        print(f"\n[{idx}/Pool] Testing Vault Account [{t_path.stem}]...")
        load_vault_token(t_path)
        success, reason = try_create_session(session_name=session_name, gpu=gpu)
        if success:
            print(f"🎉 SUCCESS! T4 GPU allocated using Vault Account [{t_path.stem}]!")
            return True
        print(f"⚠️ Vault Account [{t_path.stem}] unavailable: {reason}")

    print("\n❌ All saved Google Accounts in Vault hit GPU rate-limits.")
    return False


def smart_account_manager(session_name: str = "t4-prod", gpu: str = "T4", auto_train: bool = True):
    print("=" * 75)
    print("⚡ AUTOMATED MULTI-ACCOUNT GPU VAULT & FAILOVER MANAGER")
    print("=" * 75)

    colab_bin = get_colab_bin()

    # 1. Try automatic failover across all stored accounts
    if auto_failover_vault_pool(session_name=session_name, gpu=gpu):
        print("\n✅ GPU Allocation Complete!")
    else:
        print("\n" + "=" * 80)
        print("🚨 ALL GOOGLE ACCOUNTS IN VAULT ARE RATE-LIMITED / QUOTA EXHAUSTED")
        print("👉 GENERATING GOOGLE OAUTH AUTHENTICATION LINK FOR NEW ACCOUNT LOGIN")
        print("=" * 80)
        print("\nInstructions:")
        print("1. Open the Google Authorization URL printed below in your browser.")
        print("2. Sign in with any new/fresh Google Account.")
        print("3. Copy the authorization code from Google and paste it into the prompt below.")
        print("4. The new account token will be permanently saved to your Vault for future runs.\n")
        
        clear_active_colab_auth()
        
        # Launch interactive CLI login for new account
        cmd = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
        proc = subprocess.run(cmd)
        
        if proc.returncode == 0 and TOKEN_FILE.exists():
            new_vault = save_active_token_to_vault()
            print(f"\n🎉 SUCCESS! New Google Account authenticated & saved to Vault: {new_vault.name if new_vault else 'Saved'}")
        else:
            print("\n❌ OAuth authentication incomplete or failed.")
            return 1

    # 2. Launch Training if requested
    if auto_train:
        print("\n" + "=" * 75)
        print("🚀 LAUNCHING QWEN3-4B PRODUCTION QLORA TRAINING")
        print("=" * 75)
        
        runner_script = Path(__file__).resolve().parent / "run_colab_job.py"
        subprocess.run([sys.executable, str(runner_script), "--action", "train", "--session", session_name, "--gpu", gpu])

    return 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Account Colab GPU Vault Manager")
    parser.add_argument("--session", "-s", default="t4-prod")
    parser.add_argument("--gpu", "-g", default="T4")
    parser.add_argument("--list-vault", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    if args.list_vault:
        tokens = get_vault_tokens()
        print(f"\n📁 Saved Account Tokens in Vault ({len(tokens)}):")
        for i, t in enumerate(tokens, 1):
            print(f"   [{i}] {t.name}")
        return 0

    return smart_account_manager(session_name=args.session, gpu=args.gpu, auto_train=not args.no_train)


if __name__ == "__main__":
    sys.exit(main())
