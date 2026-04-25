"""Convenience wrapper for starting the sre-gym FastAPI server.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --port 8000
    python scripts/run_server.py --host 0.0.0.0

The server exposes:

    POST /reset    POST /step    GET /state          (OpenEnv contract)
    GET  /tasks    GET  /baseline GET /grader         (sre-gym extensions)
    GET  /status   GET  /health
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from unified_incident_env.server.app import create_compatible_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--workers", type=int, default=1, help="ignored when --reload is set")
    parser.add_argument("--reload", action="store_true", help="enable auto-reload")
    args = parser.parse_args()

    if args.reload:
        uvicorn.run(
            "unified_incident_env.server.app:create_compatible_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        app = create_compatible_app()
        uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
