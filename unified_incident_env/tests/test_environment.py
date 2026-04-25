"""Behavior and API tests for the honest narrow incident environment."""

from __future__ import annotations

from fastapi.testclient import TestClient

from unified_incident_env.models import HypothesisPayload, UnifiedIncidentAction
from unified_incident_env.server import app as app_module
from unified_incident_env.server.challenge import DEFAULT_SCENARIO_ID, SCENARIOS, list_baselines, scenario_for_difficulty
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
    assert obs.final_score > 0.55


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
    scenarios_by_difficulty = {scenario["difficulty"] for scenario in payload["scenarios"]}
    assert {"easy", "medium", "hard"}.issubset(scenarios_by_difficulty)
    assert {"easy", "medium", "hard"}.issubset(set(payload["available_difficulties"]))

    baseline = client.get("/baseline")
    assert baseline.status_code == 200
    baseline_payload = baseline.json()
    baseline_ids = {item["scenario_id"] for item in baseline_payload["baselines"]}
    assert {"worker_deploy_cascade", "db_config_rollout", "gateway_auth_rollout"}.issubset(baseline_ids)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "healthy"}

    status = client.get("/status")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["progress"]["scenario_id"] == DEFAULT_SCENARIO_ID
    assert status_payload["grader"]["score"] > 0.0


def _run_baseline_for_scenario(scenario_id: str):
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=scenario_id)
    last = None
    for step in list_baselines(scenario_id).baselines[0].actions:
        last = env.step(step.action)
    return last


def test_medium_baseline_resolves_honestly() -> None:
    obs = _run_baseline_for_scenario("db_config_rollout")
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    checks = {check.name: check.passed for check in obs.checks}
    assert checks["database_recovery"] is True
    assert checks["end_to_end"] is True
    assert obs.final_score > 0.55


def test_hard_baseline_resolves_honestly() -> None:
    obs = _run_baseline_for_scenario("gateway_auth_rollout")
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    checks = {check.name: check.passed for check in obs.checks}
    assert checks["end_to_end"] is True
    assert obs.final_score > 0.55


def test_medium_wrong_rollback_target_is_penalized() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="db_config_rollout")
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="worker"))
    assert obs.reward < 0.0
    assert obs.failure_type == "wrong_remediation_target"
    assert obs.incident_resolved is False


def test_hard_wrong_rollback_target_is_penalized() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="gateway_auth_rollout")
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="worker"))
    assert obs.reward < 0.0
    assert obs.failure_type == "wrong_remediation_target"


def test_all_scenarios_expose_noise_alerts() -> None:
    env = UnifiedIncidentEnvironment()
    for scenario_id in (
        "worker_deploy_cascade",
        "db_config_rollout",
        "gateway_auth_rollout",
        "payment_webhook_misconfig",
        "schema_drift_missing_migration",
        "cache_stale_state",
    ):
        obs = env.reset(scenario_id=scenario_id)
        assert len(obs.noise_alerts) > 0, f"{scenario_id} should expose noise_alerts"
        assert all(alert.message for alert in obs.noise_alerts)


def test_blast_radius_increments_on_mitigations() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="worker_deploy_cascade")
    obs0 = env.step(UnifiedIncidentAction(action_type="query_logs", service="worker"))
    assert obs0.blast_radius == 0
    env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="worker"))
    obs2 = env.step(UnifiedIncidentAction(action_type="restart_service", service="database"))
    assert obs2.blast_radius == 2


def test_baseline_ceiling_is_hardened_below_080() -> None:
    """Scripted-optimal baseline must not score above ~0.80. Headroom left
    for a trained agent that earns speed_bonus by finishing faster than
    optimal_ticks."""
    for scenario_id in (
        "worker_deploy_cascade",
        "db_config_rollout",
        "gateway_auth_rollout",
        "payment_webhook_misconfig",
        "schema_drift_missing_migration",
        "cache_stale_state",
    ):
        obs = _run_baseline_for_scenario(scenario_id)
        assert obs is not None
        assert obs.final_score <= 0.80, f"{scenario_id} ceiling {obs.final_score} exceeds headroom budget"
        assert obs.final_score >= 0.55, f"{scenario_id} ceiling {obs.final_score} is too low; env is unsolvable"


def test_speed_bonus_rewards_finishing_under_optimal_ticks() -> None:
    """A faster solve that keeps both verification checks should beat the
    baseline ceiling by the speed_bonus margin. This is the training target
    — trained agents that skip verification to chase speed should score
    *lower*, not higher."""
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="gateway_auth_rollout")
    # 5-step path: 1 query + 1 rollback + 2 checks + 1 declare. Baseline does 8.
    env.step(UnifiedIncidentAction(action_type="query_deploys", service="api-gateway"))
    env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"))
    env.step(UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"))
    env.step(UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"))
    obs = env.step(UnifiedIncidentAction(action_type="declare_resolved"))
    assert obs.incident_resolved is True
    assert obs.score_breakdown.get("speed_bonus", 0) > 0.0
    assert obs.final_score > 0.74, f"Faster solve with full verification should beat baseline, got {obs.final_score}"


def test_hard_does_not_require_database_recovery_check() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="gateway_auth_rollout")
    env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"))
    end_to_end = env.step(UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"))
    assert any(check.name == "end_to_end" and check.passed for check in end_to_end.checks)
    resolved = env.step(UnifiedIncidentAction(action_type="declare_resolved"))
    assert resolved.incident_resolved is True


def test_procgen_catalog_registers_variants_for_each_template() -> None:
    procgen_ids = {scenario_id for scenario_id, scenario in SCENARIOS.items() if scenario.get("is_procgen")}
    assert any(scenario_id.startswith("worker_deploy_cascade__p") for scenario_id in procgen_ids)
    assert any(scenario_id.startswith("db_config_rollout__p") for scenario_id in procgen_ids)
    assert any(scenario_id.startswith("gateway_auth_rollout__p") for scenario_id in procgen_ids)
    assert any(scenario_id.startswith("payment_webhook_misconfig__p") for scenario_id in procgen_ids)
    assert any(scenario_id.startswith("schema_drift_missing_migration__p") for scenario_id in procgen_ids)
    assert any(scenario_id.startswith("cache_stale_state__p") for scenario_id in procgen_ids)


def test_payment_webhook_baseline_resolves_honestly() -> None:
    obs = _run_baseline_for_scenario("payment_webhook_misconfig")
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    assert obs.final_score > 0.55


def test_payment_webhook_wrong_rollback_target_is_penalized() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="payment_webhook_misconfig")
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="worker"))
    assert obs.reward < 0.0
    assert obs.failure_type == "wrong_remediation_target"


def test_schema_drift_baseline_resolves_honestly() -> None:
    obs = _run_baseline_for_scenario("schema_drift_missing_migration")
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    assert obs.final_score > 0.55


def test_schema_drift_wrong_rollback_target_is_penalized() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="schema_drift_missing_migration")
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="database"))
    assert obs.reward < 0.0
    assert obs.failure_type == "wrong_remediation_target"


def test_cache_stale_state_baseline_resolves_honestly() -> None:
    obs = _run_baseline_for_scenario("cache_stale_state")
    assert obs is not None
    assert obs.done is True
    assert obs.incident_resolved is True
    assert obs.final_score > 0.55


def test_cache_stale_state_requires_cache_rollback() -> None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id="cache_stale_state")
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"))
    assert obs.reward < 0.0
    assert obs.failure_type == "wrong_remediation_target"


def test_catalog_has_twelve_scenario_templates() -> None:
    templates = {
        scenario_id
        for scenario_id, scenario in SCENARIOS.items()
        if not scenario.get("is_procgen")
    }
    assert templates == {
        # v2 templates (kept verbatim)
        "worker_deploy_cascade",
        "db_config_rollout",
        "gateway_auth_rollout",
        "payment_webhook_misconfig",
        "schema_drift_missing_migration",
        "cache_stale_state",
        # round-2 templates (added for OpenEnv April 2026 submission;
        # see docs/BASIC_TIER.md for the per-template skill mapping)
        "dep_degradation",
        "memory_leak_oom",
        "auth_token_expiry",
        "network_partition",
        "rate_limit_retry_storm",
        "migration_lock",
    }


def test_scenario_for_difficulty_seed_is_deterministic() -> None:
    first = scenario_for_difficulty("medium", seed=7)
    second = scenario_for_difficulty("medium", seed=7)
    assert first["id"] == second["id"]
    assert first["difficulty"] == "medium"


def test_procgen_variant_baseline_routes_through_template_builder() -> None:
    scenario_id = next(
        current_id
        for current_id, scenario in SCENARIOS.items()
        if scenario.get("is_procgen") and scenario.get("template_id") == "db_config_rollout"
    )
    obs = _run_baseline_for_scenario(scenario_id)
    assert obs is not None
    assert obs.incident_resolved is True
    assert obs.final_score >= 0.55


def test_noise_service_queries_are_scored_as_noise() -> None:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id="gateway_auth_rollout__p01")
    noise_service = obs.noise_alerts[0].service
    noise_obs = env.step(UnifiedIncidentAction(action_type="query_logs", service=noise_service))
    assert noise_obs.noise_queries == 1
    assert noise_service in (noise_obs.tool_output or "")
    assert noise_obs.score_breakdown["noise_handling_score"] < 0.05
