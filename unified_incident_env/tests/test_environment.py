"""Behavior and API tests for the honest narrow incident environment."""

from __future__ import annotations

from fastapi.testclient import TestClient

from unified_incident_env.models import HypothesisPayload, UnifiedIncidentAction
from unified_incident_env.server import app as app_module
from unified_incident_env.server.challenge import DEFAULT_SCENARIO_ID, list_baselines
from unified_incident_env.server.environment import UnifiedIncidentEnvironment


def _run_baseline(env: UnifiedIncidentEnvironment):
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    last = None
    for step in list_baselines(DEFAULT_SCENARIO_ID).baselines[0].actions:
        last = env.step(step.action)
    return last


def test_baseline_resolves_honestly() -> None:
    env = UnifiedIncidentEnvironment()
    obs = _run_baseline(env)
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    checks = {check.name: check.passed for check in obs.checks}
    assert checks["database_recovery"] is True
    assert checks["end_to_end"] is True
    assert obs.final_score > 0.7


def test_query_deploys_reveals_evidence_but_not_positive_reward() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    obs = env.step(UnifiedIncidentAction(action_type="query_deploys", service="worker"))
    assert obs.reward <= 0.0
    assert "worker@2026.04.23-bad" in (obs.tool_output or "")
    assert obs.incident_resolved is False


def test_restart_database_before_rollback_is_negative() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    obs = env.step(UnifiedIncidentAction(action_type="restart_service", service="database"))
    assert obs.reward < 0.0
    assert obs.failure_type == "premature_restart"
    assert obs.incident_resolved is False
    assert obs.service_health["database"].status == "crashed"


def test_duplicate_hypothesis_bonus_is_not_farmable() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    action = UnifiedIncidentAction(
        action_type="submit_hypothesis",
        hypothesis=HypothesisPayload(
            root_cause="bad_worker_deploy",
            affected_services=["worker", "database", "api-gateway"],
            confidence=0.82,
            recommended_next_action="rollback_deploy",
        ),
    )
    first = env.step(action)
    second = env.step(action)
    assert first.reward > second.reward
    assert second.reward <= 0.0


def test_isolating_worker_contains_but_does_not_resolve() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    isolated = env.step(UnifiedIncidentAction(action_type="isolate_service", service="worker"))
    assert isolated.containment_applied is True
    assert isolated.incident_resolved is False
    checked = env.step(UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"))
    checks = {check.name: check.passed for check in checked.checks}
    assert checks["end_to_end"] is False


def test_declare_resolved_requires_checks() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    obs = env.step(UnifiedIncidentAction(action_type="declare_resolved"))
    assert obs.reward < 0.0
    assert obs.done is False
    assert obs.failure_type == "premature_resolution"


def test_observation_exposes_bounded_actions_without_valid_example() -> None:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=DEFAULT_SCENARIO_ID)
    assert obs.allowed_actions == [
        "query_logs",
        "query_metrics",
        "query_dependencies",
        "query_deploys",
        "rollback_deploy",
        "restart_service",
        "run_check",
        "isolate_service",
        "escalate",
        "submit_hypothesis",
        "declare_resolved",
    ]
    assert obs.valid_action_example is None


def test_routes_expose_new_catalog_and_status(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    client = TestClient(app_module.create_compatible_app())

    tasks = client.get("/tasks")
    assert tasks.status_code == 200
    payload = tasks.json()
    assert payload["default_scenario_id"] == DEFAULT_SCENARIO_ID
    assert len(payload["scenarios"]) == 1

    baseline = client.get("/baseline")
    assert baseline.status_code == 200
    baseline_payload = baseline.json()
    assert baseline_payload["baselines"][0]["scenario_id"] == DEFAULT_SCENARIO_ID

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "healthy"}

    status = client.get("/status")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["progress"]["scenario_id"] == DEFAULT_SCENARIO_ID
    assert status_payload["grader"]["score"] > 0.0
