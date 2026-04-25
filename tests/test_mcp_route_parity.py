"""Parity test: every action reachable via /step is reachable via /mcp.

The /mcp route exposes the same 11 actions as JSON-RPC 2.0 ``tools/call``
methods. For each action, this test:

1. Asserts the tool descriptor exists under ``tools/list``.
2. Calls the action via /step and via /mcp on freshly-reset envs from the
   same scenario.
3. Diffs the resulting observations and asserts the load-bearing fields
   (``reward``, ``failure_type``, ``incident_resolved``, ``score_breakdown``)
   are identical.

This is the single biggest Innovation-criterion win in the round-2 sprint —
it makes the env first-class for both the OpenEnv RL training loop *and*
MCP-aware LLM agents (Claude / Cursor / etc.).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from unified_incident_env.server.app import create_compatible_app
from unified_incident_env.server.environment import (
    ALL_ACTIONS,
    REQUIRED_FIELDS_BY_ACTION,
)
from unified_incident_env.server.mcp import build_tool_descriptors


def _client() -> TestClient:
    return TestClient(create_compatible_app())


# ---------- tools/list contract ----------


def test_tools_list_returns_eleven_tools() -> None:
    descriptors = build_tool_descriptors()
    assert len(descriptors) == 11
    assert {d["name"] for d in descriptors} == set(ALL_ACTIONS)


def test_tools_list_descriptors_have_inputschema() -> None:
    for descriptor in build_tool_descriptors():
        assert "inputSchema" in descriptor
        assert descriptor["inputSchema"]["type"] == "object"
        # Required fields list must match REQUIRED_FIELDS_BY_ACTION.
        expected_required = set(REQUIRED_FIELDS_BY_ACTION[descriptor["name"]])
        actual_required = set(descriptor["inputSchema"].get("required", []))
        assert actual_required == expected_required, (
            f"{descriptor['name']}: expected required={expected_required}, got {actual_required}"
        )


def test_tools_list_via_http() -> None:
    client = _client()
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert len(body["result"]["tools"]) == 11


def test_initialize_returns_server_info() -> None:
    client = _client()
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "initialize"},
    )
    body = resp.json()
    assert body["result"]["serverInfo"]["name"] == "sre-gym"
    assert body["result"]["protocolVersion"]


def test_ping() -> None:
    client = _client()
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "ping"})
    assert resp.json()["result"] == {"ok": True}


# ---------- /step ↔ /mcp parity ----------


# Canonical example payloads per action. Each pair must produce the same
# observation when stepped against a freshly reset env.
CANONICAL_ACTIONS: list[tuple[str, dict[str, Any]]] = [
    ("query_logs", {"service": "worker"}),
    ("query_metrics", {"service": "database", "metric": "cpu"}),
    ("query_dependencies", {"service": "api-gateway"}),
    ("query_deploys", {"service": "worker"}),
    ("rollback_deploy", {"service": "worker"}),
    ("restart_service", {"service": "database"}),  # premature_restart penalty path
    ("isolate_service", {"service": "worker"}),
    ("run_check", {"check_name": "database_recovery"}),
    (
        "submit_hypothesis",
        {
            "hypothesis": {
                "root_cause": "bad_worker_deploy",
                "affected_services": ["worker", "database"],
                "confidence": 0.7,
                "recommended_next_action": "rollback_deploy",
            }
        },
    ),
    ("escalate", {}),
    ("declare_resolved", {}),     # premature_resolution path
]


@pytest.mark.parametrize("action_name,arguments", CANONICAL_ACTIONS)
def test_step_and_mcp_return_parity_observations(action_name: str, arguments: dict[str, Any]) -> None:
    """For each action, /step and /mcp must produce identical observation cores.

    The /step OpenEnv response is wrapped as
    ``{observation: {...}, reward: <r>, done: <d>}`` — reward + done live at
    the top level. /mcp returns the full observation including its own
    ``reward`` / ``done`` fields under ``result.structuredContent``.

    Parity here means: the load-bearing observation cores are identical
    once you normalize for the wrapper.
    """
    # -- /step path
    client_a = _client()
    client_a.post("/reset", json={})
    step_payload: dict[str, Any] = {"action_type": action_name, **arguments}
    step_resp = client_a.post("/step", json={"action": step_payload})
    assert step_resp.status_code == 200, step_resp.text
    step_body = step_resp.json()
    step_obs = step_body.get("observation", step_body)
    step_reward = step_body.get("reward")
    step_done = step_body.get("done")

    # -- /mcp path
    client_b = _client()
    client_b.post("/reset", json={})
    mcp_resp = client_b.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": action_name, "arguments": arguments},
        },
    )
    assert mcp_resp.status_code == 200, mcp_resp.text
    mcp_body = mcp_resp.json()
    assert "result" in mcp_body, f"MCP error: {mcp_body}"
    mcp_obs = mcp_body["result"]["structuredContent"]

    # Top-level reward/done (OpenEnv) must match the embedded reward/done (MCP).
    assert step_reward == mcp_obs.get("reward"), (
        f"{action_name}: reward mismatch — /step={step_reward!r} /mcp={mcp_obs.get('reward')!r}"
    )
    assert step_done == mcp_obs.get("done"), (
        f"{action_name}: done mismatch — /step={step_done!r} /mcp={mcp_obs.get('done')!r}"
    )

    # Observation-core fields must match (these are populated identically).
    for field in ("incident_resolved", "failure_type", "tick_count", "workflow_stage"):
        assert step_obs.get(field) == mcp_obs.get(field), (
            f"{action_name}: {field} mismatch — /step={step_obs.get(field)!r} /mcp={mcp_obs.get(field)!r}"
        )

    # Score breakdown must match (it's the rubric output, deterministic).
    assert step_obs.get("score_breakdown") == mcp_obs.get("score_breakdown"), (
        f"{action_name}: score_breakdown mismatch"
    )


def test_mcp_unknown_method_returns_jsonrpc_method_not_found() -> None:
    client = _client()
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "frobnicate"})
    body = resp.json()
    assert body["error"]["code"] == -32601


def test_mcp_unknown_tool_returns_invalid_params() -> None:
    client = _client()
    client.post("/reset", json={})
    resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "definitely_not_a_real_action", "arguments": {}},
        },
    )
    body = resp.json()
    assert body["error"]["code"] == -32602
    assert "unknown tool" in body["error"]["message"].lower()


def test_mcp_reserved_tool_names_rejected() -> None:
    """Reserved tool names (reset, step, state, close) must be rejected so MCP
    clients can't accidentally short-circuit the HTTP routes."""
    client = _client()
    client.post("/reset", json={})
    for reserved in ("reset", "step", "state", "close"):
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": reserved, "arguments": {}},
            },
        )
        body = resp.json()
        assert body["error"]["code"] == -32602
        assert "reserved" in body["error"]["message"].lower()


def test_mcp_tools_get_shortcut() -> None:
    client = _client()
    resp = client.get("/mcp/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert "tools" in body
    assert len(body["tools"]) == 11
