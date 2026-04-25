"""MCP (Model Context Protocol) JSON-RPC 2.0 dual-route for the Basic tier.

Adds a ``/mcp`` endpoint that wraps the same 11-action handlers exposed by
``/step``, but speaks the JSON-RPC 2.0 + ``tools/list`` + ``tools/call`` shape
expected by MCP clients. Every action reachable via ``/step`` is reachable via
``/mcp`` with identical return semantics — see ``tests/test_mcp_route_parity.py``.

Why this matters
----------------
The standard OpenEnv ``/step`` route is fine for RL training loops; MCP-shaped
tools are the conventional surface for LLM agents (Claude, Cursor, MCP-aware
clients). Shipping both in one server gives the env a single source-of-truth
for the action contract while letting two very different consumer classes
drive it without translation glue.

Protocol
--------
JSON-RPC 2.0 envelope::

    POST /mcp
    {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/list" | "tools/call" | "ping" | "initialize",
      "params": { ... }
    }

Methods implemented:

- ``initialize`` — returns server info + capability list. Required by MCP clients.
- ``ping`` — health probe (no params, returns ``{"ok": true}``).
- ``tools/list`` — returns the 11 actions as MCP tool descriptors. Schemas are
  derived from the ``UnifiedIncidentAction`` Pydantic model and the catalogue.
- ``tools/call`` — dispatches to the env's step handler. ``params.name`` is
  the tool name (one of the 11 actions); ``params.arguments`` is the action
  payload (e.g. ``{"service": "worker"}``). Returns the same observation
  payload ``/step`` returns, wrapped in the MCP ``content`` envelope.

Errors follow JSON-RPC 2.0 error codes:
- -32601 method not found
- -32602 invalid params
- -32603 internal error
- -32700 parse error
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models import UnifiedIncidentAction, UnifiedIncidentObservation
from .environment import ALL_ACTIONS, REQUIRED_FIELDS_BY_ACTION, UnifiedIncidentEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelope models.
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: int | str | None = None
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any = None


# ---------------------------------------------------------------------------
# MCP tool descriptors. Built once per process from the action contract.
# ---------------------------------------------------------------------------


def _service_enum() -> list[str]:
    return ["api-gateway", "cache", "database", "worker"]


def _metric_enum() -> list[str]:
    return ["cpu", "error_rate", "latency"]


def _check_enum() -> list[str]:
    return ["database_recovery", "end_to_end"]


def _hypothesis_schema() -> dict[str, Any]:
    # Imported here to avoid a hard dependency at module-import time.
    from inference import _ROOT_CAUSE_ENUM  # repo-root inference.py

    return {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string", "enum": list(_ROOT_CAUSE_ENUM)},
            "affected_services": {
                "type": "array",
                "items": {"type": "string", "enum": _service_enum()},
                "minItems": 1,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "recommended_next_action": {"type": "string", "enum": list(ALL_ACTIONS)},
        },
        "required": ["root_cause", "affected_services", "confidence", "recommended_next_action"],
        "additionalProperties": False,
    }


_ACTION_DESCRIPTIONS: dict[str, str] = {
    "query_logs": "Read the log stream for one service (api-gateway, cache, database, worker).",
    "query_metrics": "Read a metric (cpu / error_rate / latency) for one service.",
    "query_dependencies": "Read the upstream/downstream dependency chain for one service.",
    "query_deploys": "Read recent deploy history for one service (returns deploy version + relative timestamp).",
    "rollback_deploy": "Revert the most recent deploy on one service. Negative reward if wrong target.",
    "restart_service": "Restart one service. Rejected with `failure_type=premature_restart` if the cause hasn't been removed first.",
    "isolate_service": "Cordon one service. Applies containment but does not on its own resolve the incident.",
    "run_check": "Run an explicit verification check (database_recovery or end_to_end).",
    "submit_hypothesis": "Submit a structured hypothesis about the root cause. Idempotent (anti-gaming).",
    "escalate": "No-op step that records that human attention was requested.",
    "declare_resolved": "Terminal action. Rejected with `failure_type=premature_resolution` if the resolution check hasn't passed.",
}


def _tool_input_schema(action: str) -> dict[str, Any]:
    """Build a JSON Schema for the arguments of a single MCP tool call."""
    properties: dict[str, Any] = {}
    required = list(REQUIRED_FIELDS_BY_ACTION[action])
    for field in REQUIRED_FIELDS_BY_ACTION[action]:
        if field == "service":
            properties["service"] = {"type": "string", "enum": _service_enum()}
        elif field == "metric":
            properties["metric"] = {"type": "string", "enum": _metric_enum()}
        elif field == "check_name":
            properties["check_name"] = {"type": "string", "enum": _check_enum()}
        elif field == "hypothesis":
            properties["hypothesis"] = _hypothesis_schema()
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_tool_descriptors() -> list[dict[str, Any]]:
    """Build the MCP tool catalogue served by ``tools/list``.

    Mirrors the 11 actions exposed by ``/step``. Action_type is implicit (it's
    the tool name), so the input schema only requires the auxiliary fields.
    """
    descriptors = []
    for action in ALL_ACTIONS:
        descriptors.append({
            "name": action,
            "description": _ACTION_DESCRIPTIONS[action],
            "inputSchema": _tool_input_schema(action),
        })
    return descriptors


# ---------------------------------------------------------------------------
# Server-side request dispatch.
# ---------------------------------------------------------------------------


def _ok_response(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error_response(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _wrap_observation(obs: UnifiedIncidentObservation) -> dict[str, Any]:
    """Wrap an env observation in the MCP ``content`` envelope.

    MCP tools return ``content: list[ContentBlock]`` plus a structured
    ``structuredContent`` field for clients that prefer the typed surface.
    """
    obs_dict = obs.model_dump(mode="json")
    text_summary = (
        f"reward={obs_dict.get('reward', 0.0):+.3f} "
        f"score={obs_dict.get('final_score', 0.0):.3f} "
        f"resolved={obs_dict.get('incident_resolved', False)} "
        f"done={obs_dict.get('done', False)}"
    )
    return {
        "content": [
            {"type": "text", "text": text_summary},
        ],
        "structuredContent": obs_dict,
        "isError": bool(obs_dict.get("failure_type")),
    }


def _handle_method(env: UnifiedIncidentEnvironment, method: str, params: Any) -> Any:
    """Dispatch a JSON-RPC method against the active env instance."""
    if method == "initialize":
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "sre-gym", "version": "3.0.0"},
        }
    if method == "ping":
        return {"ok": True}
    if method == "tools/list":
        return {"tools": build_tool_descriptors()}
    if method == "tools/call":
        if not isinstance(params, dict):
            raise ValueError("tools/call requires object params")
        name = params.get("name")

        # Reject reserved tool names BEFORE the unknown-tool check, so the
        # error message points the caller at the correct HTTP route.
        if name in {"reset", "step", "state", "close"}:
            raise ValueError(f"reserved tool name {name!r}; use /{name} HTTP route instead")

        if name not in ALL_ACTIONS:
            raise ValueError(f"unknown tool {name!r}; valid: {', '.join(ALL_ACTIONS)}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")

        # Build a UnifiedIncidentAction from name + arguments.
        action_payload: dict[str, Any] = {"action_type": name, **arguments}
        action = UnifiedIncidentAction(**action_payload)
        observation = env.step(action)
        return _wrap_observation(observation)
    raise LookupError(f"method not found: {method}")


def _process_request_payload(env: UnifiedIncidentEnvironment, payload: dict[str, Any]) -> dict[str, Any]:
    """Process a single JSON-RPC 2.0 request payload and return the response."""
    try:
        req = JsonRpcRequest(**payload)
    except Exception as exc:
        return _error_response(payload.get("id"), -32600, f"invalid request: {exc}")

    try:
        result = _handle_method(env, req.method, req.params)
    except LookupError as exc:
        return _error_response(req.id, -32601, str(exc))
    except (ValueError, TypeError) as exc:
        return _error_response(req.id, -32602, f"invalid params: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("mcp internal error")
        return _error_response(req.id, -32603, f"internal error: {exc}")

    return _ok_response(req.id, result)


# ---------------------------------------------------------------------------
# FastAPI route attachment.
# ---------------------------------------------------------------------------


EnvFactory = Callable[[], UnifiedIncidentEnvironment]


def attach_mcp_routes(app: FastAPI, env_factory: EnvFactory) -> None:
    """Mount the ``/mcp`` JSON-RPC 2.0 dual-route on the given FastAPI app.

    OpenEnv-core ships its own placeholder ``/mcp`` POST route that returns
    "Environment does not support MCP" — we override it by removing that route
    before registering our richer JSON-RPC 2.0 dispatch.
    """
    # Single shared env instance for the lifetime of the app — the same pattern
    # the OpenEnv /step route uses. Each MCP request mutates the same env, so
    # callers should /reset between independent episodes.
    shared_env = env_factory()

    # Strip openenv-core's placeholder /mcp routes if present.
    app.router.routes = [
        r for r in app.router.routes
        if not (getattr(r, "path", None) == "/mcp"
                and "POST" in (getattr(r, "methods", None) or set()))
    ]

    @app.post("/mcp", tags=["mcp"], summary="MCP JSON-RPC 2.0 endpoint")
    async def mcp_handler(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            return JSONResponse(_error_response(None, -32700, f"parse error: {exc}"), status_code=400)

        # Single request OR batch (array of requests)
        if isinstance(payload, list):
            responses = [_process_request_payload(shared_env, item) for item in payload]
            return JSONResponse(responses)
        if isinstance(payload, dict):
            response = _process_request_payload(shared_env, payload)
            status = 200 if "result" in response else 200
            return JSONResponse(response, status_code=status)
        return JSONResponse(_error_response(None, -32600, "request must be object or array"), status_code=400)

    @app.get("/mcp/tools", tags=["mcp"], summary="MCP tools/list (GET shortcut)")
    def mcp_tools_get() -> dict[str, Any]:
        """GET shortcut for ``tools/list`` — convenient for browser inspection."""
        return {"tools": build_tool_descriptors()}

    @app.post("/mcp/reset", tags=["mcp"], summary="Reset the shared MCP env")
    async def mcp_reset(request: Request) -> dict[str, Any]:
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        scenario_id = body.get("scenario_id")
        difficulty = body.get("difficulty")
        kwargs: dict[str, Any] = {}
        if scenario_id:
            kwargs["scenario_id"] = scenario_id
        if difficulty:
            kwargs["difficulty"] = difficulty
        observation = shared_env.reset(**kwargs)
        return _wrap_observation(observation)
