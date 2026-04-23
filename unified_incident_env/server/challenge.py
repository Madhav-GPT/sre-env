"""Scenario catalog, baselines, and runtime helpers for the honest v2 core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..models import (
    BaselineCatalog,
    BaselineDefinition,
    BaselineStep,
    ScenarioCatalog,
    ScenarioSummary,
    UnifiedIncidentAction,
)

DEFAULT_SCENARIO_ID = "worker_deploy_cascade"

SCENARIOS: dict[str, dict[str, Any]] = {
    "worker_deploy_cascade": {
        "id": "worker_deploy_cascade",
        "difficulty": "easy",
        "name": "Worker Deploy Cascade",
        "description": (
            "A bad worker deploy causes sustained database overload and login 502s at the gateway. "
            "The agent must diagnose from evidence, choose a safe remediation, verify recovery, and declare resolved only after checks pass."
        ),
        "root_cause": "A bad worker deploy is driving repeated database overload.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.4,
            "database": 0.4,
            "api-gateway": 0.2,
            "cache": 0.0,
        },
        "reward_config": {
            "step_cost": 0.01,
            "redundant_action_penalty": 0.02,
            "unsafe_action_penalty": 0.08,
            "premature_resolution_penalty": 0.2,
            "successful_resolution_bonus": 0.25,
            "hypothesis_bonus_scale": 0.12,
            "forbidden_reward_sources": [
                "evidence_discovery",
                "query_success",
                "unlock_events",
                "stage_advancement",
                "patch_id_selection",
            ],
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 61.0,
                "memory_pct": 38.0,
                "error_rate_pct": 24.0,
                "latency_ms": 640.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 18.0,
                "memory_pct": 24.0,
                "error_rate_pct": 0.0,
                "latency_ms": 14.0,
            },
            "database": {
                "status": "crashed",
                "cpu_pct": 99.0,
                "memory_pct": 97.0,
                "error_rate_pct": 100.0,
                "latency_ms": 0.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 88.0,
                "memory_pct": 71.0,
                "error_rate_pct": 19.0,
                "latency_ms": 420.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Login requests are returning sustained 502s.",
            },
            {
                "service": "database",
                "severity": "critical",
                "message": "Database process is crashing under repeated overload.",
            },
            {
                "service": "worker",
                "severity": "warning",
                "message": "Worker queue depth and retry volume spiked after a recent rollout.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway upstream errors point to worker timeouts followed by database connection failures. "
                "No recent gateway deploys are recorded."
            ),
            "cache": "Cache hit ratio is stable and cache upstream probes remain healthy.",
            "database": (
                "Database logs show repeated bursts of expensive worker-originated writes immediately before each crash."
            ),
            "worker": (
                "Worker logs show request fanout amplification and elevated retries beginning right after rollout build worker@2026.04.23-bad."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 502 rate is 24% and closely tracks worker timeout bursts.",
                "latency": "Gateway p95 latency climbed to 640ms while waiting on downstream worker/database calls.",
            },
            "database": {
                "cpu": "Database CPU is pinned at 99% until the process exits.",
                "latency": "Database latency spikes sharply before each crash loop.",
            },
            "worker": {
                "cpu": "Worker CPU is 88% with growing queue pressure.",
                "error_rate": "Worker retry/error rate is elevated after rollout.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database",
            "worker": "worker -> database",
            "database": "database is a terminal dependency for write-heavy worker jobs",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache": "No cache deploys in the last 24h.",
            "database": "No database deploys in the last 24h.",
            "worker": "Rolled out worker@2026.04.23-bad 12 minutes ago.",
        },
        "checks": {
            "database_recovery": "Confirms the database is healthy and no longer crashing.",
            "end_to_end": "Confirms login traffic succeeds without worker-induced overload.",
        },
        "truth": {
            "root_cause": "bad_worker_deploy",
            "affected_services": ["worker", "database", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
    }
}

_RUNTIME_PROGRESS: dict[str, Any] | None = None


def get_scenario(scenario_id: str) -> dict[str, Any]:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id {scenario_id!r}")
    return deepcopy(SCENARIOS[scenario_id])


def scenario_for_difficulty(difficulty: str) -> dict[str, Any]:
    for scenario in SCENARIOS.values():
        if scenario["difficulty"] == difficulty:
            return deepcopy(scenario)
    raise ValueError(f"Unknown difficulty {difficulty!r}")


def list_scenarios(difficulty: str | None = None) -> ScenarioCatalog:
    if difficulty is not None and difficulty != "easy":
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    scenarios = [
        ScenarioSummary(
            id=scenario["id"],
            difficulty=scenario["difficulty"],
            name=scenario["name"],
            description=scenario["description"],
            root_cause=scenario["root_cause"],
            optimal_ticks=scenario["optimal_ticks"],
        )
        for scenario in SCENARIOS.values()
        if difficulty is None or scenario["difficulty"] == difficulty
    ]
    return ScenarioCatalog(
        default_scenario_id=DEFAULT_SCENARIO_ID,
        available_difficulties=["easy"],
        filtered_difficulty=difficulty,
        scenarios=scenarios,
    )


def _baseline_actions(scenario_id: str) -> list[BaselineStep]:
    if scenario_id != DEFAULT_SCENARIO_ID:
        raise ValueError(f"No baseline for scenario_id {scenario_id!r}")
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="worker"),
            rationale="Check whether any recent deploy aligns with the incident start.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="worker"),
            rationale="Inspect worker logs because deploy timing and queue pressure suggest worker-originated harm.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_metrics", service="database", metric="cpu"),
            rationale="Confirm that the database is overloaded as a downstream effect.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_dependencies", service="api-gateway"),
            rationale="Verify the gateway depends on the worker and database path.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "bad_worker_deploy",
                    "affected_services": ["worker", "database", "api-gateway"],
                    "confidence": 0.82,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Commit a calibrated hypothesis before taking an invasive mitigation step.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="worker"),
            rationale="Remove the triggering change before restarting downstream services.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="restart_service", service="database"),
            rationale="Bring the database back cleanly after the root cause is removed.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Verify the database is no longer crashing.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify gateway traffic succeeds end-to-end.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def list_baselines(scenario_id: str | None = None) -> BaselineCatalog:
    scenario_ids = [scenario_id] if scenario_id is not None else [DEFAULT_SCENARIO_ID]
    baselines = [
        BaselineDefinition(
            scenario_id=current_id,
            name="deterministic-remediation-baseline",
            description="Minimal honest baseline that diagnoses from evidence, rolls back the worker, restarts the database, verifies recovery, and then declares resolved.",
            optimal_ticks=SCENARIOS[current_id]["optimal_ticks"],
            actions=_baseline_actions(current_id),
        )
        for current_id in scenario_ids
    ]
    return BaselineCatalog(baselines=baselines)


def set_runtime_progress(progress: dict[str, Any]) -> None:
    global _RUNTIME_PROGRESS
    _RUNTIME_PROGRESS = deepcopy(progress)


def current_runtime_progress() -> dict[str, Any]:
    if _RUNTIME_PROGRESS is None:
        raise ValueError("Runtime progress is not initialized")
    return deepcopy(_RUNTIME_PROGRESS)


def grade_episode(state: dict[str, Any]):
    from .grader import UnifiedIncidentGrader

    scenario_id = state.get("scenario_id", DEFAULT_SCENARIO_ID)
    return UnifiedIncidentGrader().build_report(state, get_scenario(scenario_id))
