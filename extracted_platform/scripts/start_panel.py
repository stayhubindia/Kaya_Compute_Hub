#!/usr/bin/env python3
"""
Launcher script for Qwen AI Studio & QLoRA Mission Control Panel.
Usage:
    python scripts/start_panel.py [--port 7860]
"""

import argparse
import os
import socket
import sys
import webbrowser
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.panel.server import create_app
from aiohttp import web


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Launch Qwen AI Studio Mission Control Web Panel")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    port = args.port
    if is_port_in_use(port):
        new_port = find_free_port(port + 1)
        print(f"[Notice] Port {port} is already in use. Switching to port {new_port}.")
        port = new_port

    url = f"http://localhost:{port}"
    print("=" * 80)
    print("🚀 QWEN AI STUDIO & QLORA MISSION CONTROL WEB PANEL")
    print(f"🔗 Dashboard URL: {url}")
    print("=" * 80)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
