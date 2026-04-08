from __future__ import annotations

import json

import inference
from unified_incident_env.models import (
    Alert,
    SecurityContext,
    ServiceHealth,
    UnifiedIncidentObservation,
)


def make_observation(**overrides: object) -> UnifiedIncidentObservation:
    defaults = {
        "prompt_text": "Test prompt",
        "tick_count": 0,
        "max_ticks": 10,
        "difficulty": "easy",
        "workflow_stage": "diagnosis",
        "active_alerts": [
            Alert(service="database", severity="critical", message="database down")
        ],
        "service_health": {
            "api-gateway": ServiceHealth(
                name="api-gateway",
                status="degraded",
                cpu_pct=50.0,
                memory_pct=40.0,
                error_rate_pct=10.0,
                latency_ms=100.0,
            ),
            "cache": ServiceHealth(
                name="cache",
                status="healthy",
                cpu_pct=20.0,
                memory_pct=20.0,
                error_rate_pct=0.0,
                latency_ms=10.0,
            ),
            "database": ServiceHealth(
                name="database",
                status="crashed",
                cpu_pct=99.0,
                memory_pct=99.0,
                error_rate_pct=100.0,
                latency_ms=0.0,
            ),
            "worker": ServiceHealth(
                name="worker",
                status="healthy",
                cpu_pct=15.0,
                memory_pct=20.0,
                error_rate_pct=0.0,
                latency_ms=12.0,
            ),
        },
        "last_action_result": "",
        "tool_output": None,
        "failure_type": None,
        "why_failed": None,
        "allowed_actions": ["query_logs", "query_metrics", "query_dependencies"],
        "required_fields_by_action": {
            "query_logs": ["service"],
            "query_metrics": ["service", "metric"],
            "query_dependencies": ["service"],
        },
        "valid_action_example": {"action_type": "query_logs", "service": "database"},
        "common_trap": None,
        "loop_warning": None,
        "blocked_until_security_complete": False,
        "security_unlock_reason": None,
        "best_recovery_action_family": None,
        "progress_flags": {},
        "security_subquest_status": "locked",
        "security_context": SecurityContext(),
        "final_score": 0.1,
        "score_breakdown": {"efficiency_score": 0.1},
        "incident_resolved": False,
        "reward": 0.0,
        "done": False,
    }
    defaults.update(overrides)
    return UnifiedIncidentObservation(**defaults)


def test_log_helpers_match_required_format(capsys) -> None:
    inference.log_start(task="database_sqli_outage", env="unified-incident-env", model="demo-model")
    inference.log_step(step=2, action='{"action_type":"query_logs","service":"database"}', reward=0.1, done=False, error=None)
    inference.log_end(success=True, steps=2, score=0.37, rewards=[0.1, 0.27])
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == [
        "[START] task=database_sqli_outage env=unified-incident-env model=demo-model",
        '[STEP] step=2 action={"action_type":"query_logs","service":"database"} reward=0.10 done=false error=null',
        "[END] success=true steps=2 score=0.37 rewards=0.10,0.27",
    ]


def test_parse_action_strips_extra_keys_and_normalizes_aliases() -> None:
    observation = make_observation(
        workflow_stage="security_subquest",
        allowed_actions=["classify_vulnerability", "apply_patch"],
        required_fields_by_action={
            "classify_vulnerability": ["vulnerability_type"],
            "apply_patch": ["patch_id"],
        },
        valid_action_example={
            "action_type": "classify_vulnerability",
            "vulnerability_type": "sql_injection",
        },
    )
    raw = json.dumps(
        {
            "action": "classify_vulnerability",
            "vulnerability": "sql_injection",
            "details": "ignored",
        }
    )
    action = inference.parse_action(raw, observation)
    assert action == inference.UnifiedIncidentAction(
        action_type="classify_vulnerability",
        vulnerability_type="sql_injection",
    )


def test_parse_action_accepts_bare_action_using_stage_example() -> None:
    observation = make_observation(
        workflow_stage="security_subquest",
        allowed_actions=["apply_patch"],
        required_fields_by_action={"apply_patch": ["patch_id"]},
        valid_action_example={"action_type": "apply_patch", "patch_id": "parameterized_query"},
    )
    action = inference.parse_action('"apply_patch"', observation)
    assert action == inference.UnifiedIncidentAction(
        action_type="apply_patch",
        patch_id="parameterized_query",
    )


def test_build_fallback_action_uses_public_stage_signals_only() -> None:
    observation = make_observation(
        workflow_stage="security_subquest",
        allowed_actions=[
            "inspect_code",
            "classify_vulnerability",
            "apply_patch",
            "verify_security_fix",
            "submit_security_fix",
        ],
        required_fields_by_action={
            "inspect_code": [],
            "classify_vulnerability": ["vulnerability_type"],
            "apply_patch": ["patch_id"],
            "verify_security_fix": [],
            "submit_security_fix": [],
        },
        security_subquest_status="active",
        security_context=SecurityContext(code_visible=True),
        tool_output="Patch options: parameterized_query, strip_quotes, disable_login",
        prompt_text="The login SQL query is vulnerable to SQL injection.",
    )
    action = inference.build_fallback_action(observation, history=[])
    assert action == inference.UnifiedIncidentAction(
        action_type="classify_vulnerability",
        vulnerability_type="sql_injection",
    )


def test_build_policy_card_is_compact_and_stage_aware() -> None:
    observation = make_observation(
        workflow_stage="diagnosis",
        allowed_actions=["query_logs", "query_metrics", "query_dependencies"],
        valid_action_example={"action_type": "query_logs", "service": "database"},
    )
    state = inference.PolicyCardState(
        schema_notes=[
            inference.PolicyNote(
                stage="diagnosis",
                failure_type="invalid_model_output",
                mistake="Previous output was not valid JSON.",
                correction="Return exactly one JSON object.",
                valid_example={"action_type": "query_logs", "service": "database"},
            )
        ]
    )
    card = inference.build_policy_card(observation, state)
    assert "STAGE: diagnosis" in card
    assert "GOAL: find the most relevant next investigation step" in card
    assert '"action_type":"query_logs"' in card


def test_build_user_prompt_uses_policy_card_not_recent_history() -> None:
    observation = make_observation()
    prompt = inference.build_user_prompt(observation, "Session policy card:\n- Return JSON only.")
    assert "Session policy card:" not in prompt
    assert "RECENT_HISTORY" not in prompt
    assert "Current stage: diagnosis" in prompt
    assert "Current goal: find the most relevant next investigation step" in prompt
    assert "Allowed actions:" in prompt
    assert "Required fields:" in prompt
    assert "Final score: 0.1000" in prompt
    assert "Valid example:" in prompt


def test_build_user_prompt_includes_full_security_code_context() -> None:
    observation = make_observation(
        workflow_stage="security_subquest",
        allowed_actions=["classify_vulnerability", "apply_patch"],
        required_fields_by_action={
            "classify_vulnerability": ["vulnerability_type"],
            "apply_patch": ["patch_id"],
        },
        security_context=SecurityContext(code_visible=True),
        tool_output=(
            "def build_export(filename):\n"
            "    cmd = '/usr/bin/zip /tmp/out.zip ' + filename\n"
            "    return subprocess.check_output(cmd, shell=True)\n"
            "Patch options: avoid_shell, sanitize_quotes_only, disable_worker_commands"
        ),
        valid_action_example={
            "action_type": "classify_vulnerability",
            "vulnerability_type": "command_injection",
        },
    )
    prompt = inference.build_user_prompt(observation, "")
    assert "Tool output:" in prompt
    assert "subprocess.check_output" in prompt
    assert "shell=True" in prompt


def test_build_user_prompt_small_mode_extracts_policy_card_correction(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_MODE", "small")
    observation = make_observation()
    prompt = inference.build_user_prompt(
        observation,
        "LESSON: the previous action was wrong; do not repeat it",
    )
    assert "Previous action correction:" in prompt
    assert "the previous action was wrong; do not repeat it" in prompt
    assert "LESSON:" not in prompt


def test_inference_mode_defaults_to_judge(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_MODE", raising=False)
    assert inference._inference_mode() == "judge"


def test_hard_scenario_switches_to_dependency_bridge_after_worker_evidence() -> None:
    observation = make_observation(
        workflow_stage="root_cause_analysis",
        security_subquest_status="locked",
        allowed_actions=["query_logs", "query_metrics", "query_dependencies"],
        valid_action_example={"action_type": "query_logs", "service": "worker"},
    )
    history = [
        {
            "action": inference.UnifiedIncidentAction(
                action_type="query_logs",
                service="worker",
            )
        }
    ]
    narrowed = inference._narrow_allowed_actions(
        observation,
        scenario_id="worker_bad_deploy_command_injection",
        history=history,
    )
    assert narrowed == ["query_dependencies"]


def test_hard_scenario_forces_worker_rollback_after_security_completion() -> None:
    observation = make_observation(
        workflow_stage="remediation",
        security_subquest_status="completed",
        allowed_actions=["restart_service", "rollback_deploy"],
        service_health={
            "api-gateway": ServiceHealth(
                name="api-gateway",
                status="degraded",
                cpu_pct=50.0,
                memory_pct=40.0,
                error_rate_pct=10.0,
                latency_ms=100.0,
            ),
            "cache": ServiceHealth(
                name="cache",
                status="healthy",
                cpu_pct=20.0,
                memory_pct=20.0,
                error_rate_pct=0.0,
                latency_ms=10.0,
            ),
            "database": ServiceHealth(
                name="database",
                status="degraded",
                cpu_pct=60.0,
                memory_pct=55.0,
                error_rate_pct=15.0,
                latency_ms=120.0,
            ),
            "worker": ServiceHealth(
                name="worker",
                status="degraded",
                cpu_pct=80.0,
                memory_pct=70.0,
                error_rate_pct=20.0,
                latency_ms=200.0,
            ),
        },
    )
    narrowed = inference._narrow_allowed_actions(
        observation,
        scenario_id="worker_bad_deploy_command_injection",
        history=[],
    )
    assert narrowed == ["rollback_deploy"]
    prompt = inference.build_user_prompt(
        observation,
        "Session policy card:\n- Return JSON only.",
        scenario_id="worker_bad_deploy_command_injection",
        history=[],
    )
    assert "Important transition hint:" in prompt
    assert "rollback_deploy on worker" in prompt
