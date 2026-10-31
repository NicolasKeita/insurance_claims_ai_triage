"""Validate explicit local prerequisites without installing or running assessments."""

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"


def configuration(options):
    default = os.environ.get("CLAIMS_REVIEW_DEMO_DIR", ROOT / "artifacts/review_demo")
    demo = Path(options.demo_dir or default).resolve()
    if not (demo / "manifest.json").is_file():
        raise RuntimeError(f"Prepared demo manifest missing: {demo}. Run scripts/prepare_review_demo.py first.")
    if not all(importlib.util.find_spec(name) for name in ("uvicorn", "api")):
        raise RuntimeError("Use the project's installed .venv Python; application dependencies are required.")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js 22.12+ is required. Install Node.js, then reopen this terminal.")
    version = subprocess.check_output([node, "--version"], text=True).strip().removeprefix("v")
    if tuple(map(int, version.split(".")[:2])) < (22, 12):
        raise RuntimeError("Node.js 22.12+ is required by the frontend.")
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    if not vite.is_file():
        raise RuntimeError("Frontend dependencies missing. Run: npm --prefix frontend ci")
    if options.frontend_port == 8000 or not 1 <= options.frontend_port <= 65535:
        raise RuntimeError("Choose a frontend port between 1 and 65535 other than the API's port 8000.")
    _available_ports((8000, options.frontend_port))
    env = {**os.environ, "CLAIMS_REVIEW_DEMO": "1", "CLAIMS_REVIEW_DEMO_DIR": str(demo)}
    env.pop("DATABASE_URL", None)
    api = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", HOST, "--port", "8000"]
    frontend = [node, str(vite), "--host", HOST, "--port", str(options.frontend_port), "--strictPort"]
    return env, [
        (api, ROOT, f"http://{HOST}:8000/v1/review/demo-claims"),
        (frontend, ROOT / "frontend", f"http://{HOST}:{options.frontend_port}/"),
    ]


def _available_ports(ports):
    for port in ports:
        try:
            with socket.create_connection((HOST, port), timeout=0.3):
                raise RuntimeError(f"Port {port} is in use. Stop its server first; no existing process was changed.")
        except OSError:
            pass
