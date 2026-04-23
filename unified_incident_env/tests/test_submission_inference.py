from __future__ import annotations

import inference
from unified_incident_env.models import Alert, CheckResult, ServiceHealth, UnifiedIncidentObservation


def make_observation(**overrides: object) -> UnifiedIncidentObservation:
    defaults = {
        "prompt_text": "Honest incident prompt",
        "incident_summary": "Worker deploy is overloading the database.",
        "tick_count": 0,
        "max_ticks": 12,
        "difficulty": "easy",
        "workflow_stage": "triage",
        "active_alerts": [
            Alert(service="database", severity="critical", message="database crashing"),
            Alert(service="worker", severity="warning", message="worker retry volume elevated"),
        ],
        "service_health": {
            "api-gateway": ServiceHealth(name="api-gateway", status="degraded", cpu_pct=61.0, memory_pct=38.0, error_rate_pct=24.0, latency_ms=640.0),
            "cache": ServiceHealth(name="cache", status="healthy", cpu_pct=18.0, memory_pct=24.0, error_rate_pct=0.0, latency_ms=14.0),
            "database": ServiceHealth(name="database", status="crashed", cpu_pct=99.0, memory_pct=97.0, error_rate_pct=100.0, latency_ms=0.0),
            "worker": ServiceHealth(name="worker", status="degraded", cpu_pct=88.0, memory_pct=71.0, error_rate_pct=19.0, latency_ms=420.0),
        },
        "discovered_evidence": [],
        "recent_deploys": ["Rolled out worker@2026.04.23-bad 12 minutes ago."],
        "checks": [
            CheckResult(name="database_recovery", passed=False, detail="Database recovery has not been verified yet."),
            CheckResult(name="end_to_end", passed=False, detail="End-to-end health has not been verified yet."),
        ],
        "user_impact": 0.82,
        "slo_burn_rate": 0.91,
        "incident_resolved": False,
        "containment_applied": False,
        "last_action_result": "",
        "tool_output": None,
        "failure_type": None,
        "why_failed": None,
        "allowed_actions": [
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
        ],
        "required_fields_by_action": {
            "query_logs": ["service"],
            "query_metrics": ["service", "metric"],
            "query_dependencies": ["service"],
            "query_deploys": ["service"],
            "rollback_deploy": ["service"],
            "restart_service": ["service"],
            "run_check": ["check_name"],
            "isolate_service": ["service"],
            "escalate": [],
            "submit_hypothesis": ["hypothesis"],
            "declare_resolved": [],
        },
        "valid_action_example": None,
        "common_trap": None,
        "loop_warning": None,
        "blocked_until_security_complete": False,
        "security_unlock_reason": None,
        "best_recovery_action_family": None,
        "progress_flags": {},
        "security_subquest_status": None,
        "security_context": {},
        "final_score": 0.1,
        "score_breakdown": {"final_score": 0.1},
        "reward": 0.0,
        "done": False,
    }
    defaults.update(overrides)
    return UnifiedIncidentObservation(**defaults)


def test_log_helpers_match_required_format(capsys) -> None:
    inference.log_start(task="worker_deploy_cascade", env="unified-incident-env", model="demo-model")
    inference.log_step(step=2, action='{"action_type":"query_logs","service":"database"}', reward=-0.01, done=False, error=None)
    inference.log_end(success=True, steps=2, score=0.37, rewards=[-0.01, 0.27])
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == [
        "[START] task=worker_deploy_cascade env=unified-incident-env model=demo-model",
        '[STEP] step=2 action={"action_type":"query_logs","service":"database"} reward=-0.01 done=false error=null',
        "[END] success=true steps=2 score=0.37 rewards=-0.01,0.27",
    ]


def test_parse_action_accepts_valid_json() -> None:
    observation = make_observation()
    action = inference.parse_action('{"action_type":"query_deploys","service":"worker"}', observation)
    assert action == inference.UnifiedIncidentAction(action_type="query_deploys", service="worker")


def test_parse_action_rejects_incomplete_metric_query() -> None:
    observation = make_observation()
    assert inference.parse_action('{"action_type":"query_metrics","service":"database"}', observation) is None


def test_build_user_prompt_includes_public_state_without_examples() -> None:
    observation = make_observation()
    prompt = inference.build_user_prompt(observation)
    assert "Incident summary:" in prompt
    assert "Allowed actions:" in prompt
    assert "Required fields:" in prompt
    assert "Valid example" not in prompt
    assert "worker@2026.04.23-bad" not in prompt


def test_build_fallback_action_prefers_public_deploy_query() -> None:
    observation = make_observation()
    action = inference.build_fallback_action(observation)
    assert action == inference.UnifiedIncidentAction(action_type="query_deploys", service="worker")
