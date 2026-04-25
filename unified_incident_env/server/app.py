"""FastAPI app and metadata routes for the honest narrow incident environment."""

from __future__ import annotations

import argparse
import os
from typing import Any

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_fastapi_app

from ..models import (
    BaselineCatalog,
    GraderReport,
    RuntimeStatus,
    ScenarioCatalog,
    UnifiedIncidentAction,
    UnifiedIncidentObservation,
    UnifiedIncidentState,
)
from .challenge import current_runtime_progress, grade_episode, list_baselines, list_scenarios, set_runtime_progress
from .environment import UnifiedIncidentEnvironment
from .mcp import attach_mcp_routes

_BOOTSTRAP_ENV = UnifiedIncidentEnvironment()
set_runtime_progress(_BOOTSTRAP_ENV.state.model_dump())

_SIMPLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unified Incident Env</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }
    code, pre { background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }
    pre { padding: 12px; overflow: auto; }
  </style>
</head>
<body>
  <h1>Unified Incident Env</h1>
  <p>This v2 environment exposes an honest bounded-action incident diagnosis and remediation task.</p>
  <ul>
    <li><a href="/docs">API docs</a></li>
    <li><a href="/tasks">Scenario catalog</a></li>
    <li><a href="/baseline">Baseline plan</a></li>
    <li><a href="/status">Runtime status</a></li>
    <li><a href="/health">Health</a></li>
  </ul>
  <h2>Core ideas</h2>
  <ul>
    <li>Queries reveal evidence but do not directly mint positive reward.</li>
    <li>Remediation actions change the world state.</li>
    <li><code>run_check</code> verifies recovery explicitly.</li>
    <li><code>declare_resolved</code> succeeds only after objective checks pass.</li>
  </ul>
  <h2>Manual example</h2>
  <pre>curl -X POST http://127.0.0.1:8000/reset -H 'content-type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8000/step -H 'content-type: application/json' -d '{"action_type":"query_deploys","service":"worker"}'</pre>
</body>
</html>
"""


def create_compatible_app():
    env_factory = lambda: UnifiedIncidentEnvironment()
    app = create_fastapi_app(
        env_factory,
        UnifiedIncidentAction,
        UnifiedIncidentObservation,
        max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "32")),
    )

    # /  — reserved for the Gradio terminal UI mounted by app.py at the repo root.
    # If app.py is NOT in the import path (e.g. running uvicorn directly against
    # this module), `/` falls through to a 404; that's intentional. The legacy
    # markdown landing now lives at /info (see below).

    @app.get("/info", include_in_schema=False)
    async def web_info() -> HTMLResponse:
        """Legacy quick-links / markdown landing. Kept as a stable fallback URL."""
        return HTMLResponse(_SIMPLE_HTML)

    @app.get("/simple", include_in_schema=False)
    async def simple_console() -> HTMLResponse:
        """Backwards-compatible alias for /info — older docs / scripts may link here."""
        return HTMLResponse(_SIMPLE_HTML)

    _attach_metadata_routes(app)
    attach_mcp_routes(app, env_factory)

    return app


def _attach_metadata_routes(app):
    @app.get("/tasks", response_model=ScenarioCatalog, tags=["challenge"])
    def tasks(difficulty: str | None = None) -> ScenarioCatalog:
        try:
            return list_scenarios(difficulty=difficulty)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/baseline", response_model=BaselineCatalog, tags=["challenge"])
    def baseline(scenario_id: str | None = None) -> BaselineCatalog:
        try:
            return list_baselines(scenario_id=scenario_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/grader", response_model=GraderReport, tags=["challenge"])
    def grader(scenario_id: str | None = None) -> GraderReport:
        progress = current_runtime_progress()
        if scenario_id is not None:
            progress["scenario_id"] = scenario_id
        try:
            return grade_episode(progress)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/status", response_model=RuntimeStatus, tags=["challenge"])
    def status() -> RuntimeStatus:
        progress = current_runtime_progress()
        return RuntimeStatus(
            progress=UnifiedIncidentState(**progress),
            grader=grade_episode(progress),
        )

    @app.get("/health", tags=["challenge"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "environment": "unified_incident_env",
            "version": "2.0.0",
            "stages": ["triage", "mitigation", "validation", "resolved"],
        }


app = create_compatible_app()


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
