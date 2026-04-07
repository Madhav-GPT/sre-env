#!/usr/bin/env python3
"""Run a local end-to-end benchmark demo against the OpenEnv server."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parent
BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
HEALTH_URL = f"{BASE_URL.rstrip('/')}/health"


def server_is_ready() -> bool:
    try:
        response = httpx.get(HEALTH_URL, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def start_server() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=REPO_ROOT,
        text=True,
    )


def wait_for_server(timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_is_ready():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become ready at {HEALTH_URL}")


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    server_process: subprocess.Popen[str] | None = None
    try:
        if not server_is_ready():
            server_process = start_server()
            wait_for_server()

        env = os.environ.copy()
        env.setdefault("ENV_BASE_URL", BASE_URL)
        subprocess.run(
            [sys.executable, "inference.py"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
