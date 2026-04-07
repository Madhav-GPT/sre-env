"""Tests for the model-agnostic trainer scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from unified_incident_env.scripts.baseline_agent import plan_for_scenario
from unified_incident_env.server.challenge import get_scenario
from unified_incident_env.server.environment import UnifiedIncidentEnvironment
from unified_incident_env.trainer.action_adapter import LenientActionAdapter, StrictActionParser
from unified_incident_env.trainer.build_sft_dataset import (
    build_baseline_records,
    build_replay_records,
)
from unified_incident_env.trainer.eval_models import summarize
from unified_incident_env.trainer.prompts import build_runtime_request
from unified_incident_env.trainer.run_episode import EpisodeRunner
from unified_incident_env.trainer.trajectory_memory import CorrectionMemory, MemoryExample
from unified_incident_env.trainer.trajectory_store import TrajectoryStore
from unified_incident_env.trainer.types import EvalScenarioResult, ModelRequest, ModelResponse


class SequenceBackend:
    """Test backend that returns pre-seeded outputs in order."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.index = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        output = self.outputs[self.index]
        self.index += 1
        return ModelResponse(raw_text=output, latency_s=0.01)


def _action_jsons(scenario_id: str) -> list[str]:
    return [
        json.dumps(action.model_dump(exclude_none=True))
        for action in plan_for_scenario(scenario_id)
    ]


def test_strict_parser_rejects_extra_keys_and_missing_fields() -> None:
    parser = StrictActionParser()

    extra = parser.parse(
        '{"action_type":"query_logs","service":"database","reason":"because"}'
    )
    assert extra.parse_status == "repaired"
    assert extra.cleaned_action == {
        "action_type": "query_logs",
        "service": "database",
    }

    missing = parser.parse('{"action_type":"query_metrics","service":"database"}')
    assert missing.parse_status == "invalid_action"


def test_lenient_parser_repairs_small_schema_issues_only() -> None:
    parser = LenientActionAdapter()

    repaired = parser.parse(
        '{"action_type":"inspect_code","reason":"need context"}'
    )
    assert repaired.parse_status == "repaired"
    assert repaired.cleaned_action == {"action_type": "inspect_code"}

    metric_repaired = parser.parse(
        '{"action_type":"query_metrics","service":"database","metrics":["cpu"]}'
    )
    assert metric_repaired.parse_status == "repaired"
    assert metric_repaired.cleaned_action == {
        "action_type": "query_metrics",
        "service": "database",
        "metric": "cpu",
    }

    action_alias = parser.parse('{"action":"inspect_code"}')
    assert action_alias.parse_status == "repaired"
    assert action_alias.cleaned_action == {"action_type": "inspect_code"}

    vuln_alias = parser.parse(
        '{"action_type":"classify_vulnerability","vulnerability":"sql_injection"}'
    )
    assert vuln_alias.parse_status == "repaired"
    assert vuln_alias.cleaned_action == {
        "action_type": "classify_vulnerability",
        "vulnerability_type": "sql_injection",
    }

    bare_action = parser.parse("inspect_code")
    assert bare_action.parse_status == "repaired"
    assert bare_action.cleaned_action == {"action_type": "inspect_code"}

    invalid = parser.parse(
        '{"action_type":"classify_vulnerability","services":["cache"]}'
    )
    assert invalid.parse_status == "invalid_action"

    ambiguous = parser.parse(
        '{"action_type":"query_metrics","service":"database","metrics":["cpu","memory"]}'
    )
    assert ambiguous.parse_status == "invalid_action"


def test_episode_runner_records_successful_strict_episode() -> None:
    runner = EpisodeRunner(
        backend=SequenceBackend(_action_jsons("database_sqli_outage")),
        parser=StrictActionParser(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )

    record = runner.run("database_sqli_outage", mode="strict")
    assert record.success is True
    assert record.steps == len(plan_for_scenario("database_sqli_outage"))
    assert record.final_score > 0.5
    assert all(step.parse_status in {"ok", "repaired"} for step in record.step_records)
    assert all(step.structured_mode_used == "backend_adaptive" for step in record.step_records)


def test_episode_runner_stops_on_parse_failure() -> None:
    runner = EpisodeRunner(
        backend=SequenceBackend(["- inspect_code"]),
        parser=StrictActionParser(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )

    record = runner.run("database_sqli_outage", mode="strict")
    assert record.success is False
    assert record.failure_reason == "parse_failure:invalid_json"
    assert record.steps == 0
    assert record.step_records[0].parse_status == "invalid_json"


def test_lenient_runner_rescues_extra_keys_but_not_logic() -> None:
    outputs = []
    for payload in _action_jsons("database_sqli_outage"):
        data = json.loads(payload)
        data["reason"] = "extra explanation"
        outputs.append(json.dumps(data))

    runner = EpisodeRunner(
        backend=SequenceBackend(outputs),
        parser=LenientActionAdapter(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    record = runner.run("database_sqli_outage", mode="lenient")
    assert record.success is True
    assert any(step.parse_status == "repaired" for step in record.step_records)

    bad_runner = EpisodeRunner(
        backend=SequenceBackend(
            [
                '{"action_type":"classify_vulnerability","services":["cache"]}',
                '{"action_type":"classify_vulnerability","services":["cache"]}',
                *[
                    json.dumps(action.model_dump(exclude_none=True))
                    for action in plan_for_scenario("database_sqli_outage")[1:]
                ],
            ]
        ),
        parser=LenientActionAdapter(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    bad_record = bad_runner.run("database_sqli_outage", mode="lenient")
    assert bad_record.success is True
    assert bad_record.teacher_override_count >= 1


def test_lenient_runner_uses_repair_retry() -> None:
    plan = plan_for_scenario("database_sqli_outage")
    outputs = [
        "- invalid",
        json.dumps(plan[0].model_dump(exclude_none=True)),
        *[
            json.dumps(action.model_dump(exclude_none=True))
            for action in plan[1:]
        ],
    ]
    runner = EpisodeRunner(
        backend=SequenceBackend(outputs),
        parser=LenientActionAdapter(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    record = runner.run("database_sqli_outage", mode="lenient")
    assert record.success is True
    assert record.repair_retry_count >= 1
    assert record.step_records[0].repair_retry_used is True


def test_lenient_runner_uses_teacher_override_after_repeated_no_progress() -> None:
    outputs = [
        json.dumps({"action_type": "restart_service", "service": "database"})
        for _ in range(12)
    ]
    runner = EpisodeRunner(
        backend=SequenceBackend(outputs),
        parser=LenientActionAdapter(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    record = runner.run("database_sqli_outage", mode="lenient")
    assert record.teacher_override_count >= 1
    assert any(step.teacher_override_used for step in record.step_records)


def test_baseline_dataset_builder_emits_all_teacher_steps() -> None:
    records = build_baseline_records()
    assert len(records) == sum(
        len(plan_for_scenario(scenario_id)) for scenario_id in (
            "database_sqli_outage",
            "cache_abuse_broken_access_control",
            "worker_bad_deploy_command_injection",
        )
    )
    assert {record.source for record in records} == {"baseline"}


def test_prompt_builder_is_stage_specific() -> None:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id="database_sqli_outage")
    system_prompt, user_prompt, response_format = build_runtime_request(obs)
    assert "Return exactly one JSON object and nothing else." in system_prompt
    assert "Allowed actions:" not in system_prompt
    assert "Current stage: diagnosis" in user_prompt
    assert "Current goal: find the most relevant next investigation step" in user_prompt
    assert "Allowed actions:" in user_prompt
    assert "- query_logs" in user_prompt
    assert "- inspect_code" not in user_prompt
    assert "Final score: 0.0000" in user_prompt
    assert "Valid example:" in user_prompt
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_prompt_builder_keeps_security_code_context() -> None:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id="worker_bad_deploy_command_injection")
    obs = env.step({"action_type": "query_logs", "service": "worker"})
    obs = env.step({"action_type": "inspect_code"})
    system_prompt, user_prompt, _response_format = build_runtime_request(obs)
    assert "Return exactly one JSON object and nothing else." in system_prompt
    assert "subprocess.check_output" in user_prompt
    assert "shell=True" in user_prompt


def test_correction_memory_builds_prompt_addendum() -> None:
    strict_runner = EpisodeRunner(
        backend=SequenceBackend(["- inspect_code"]),
        parser=StrictActionParser(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    bad_record = strict_runner.run("database_sqli_outage", mode="strict")
    from unified_incident_env.trainer.analyze_failures import analyze_episode
    analysis = analyze_episode(bad_record)
    memory = CorrectionMemory()
    memory.add_episode_examples(bad_record, analysis)
    addendum = memory.build_prompt_addendum("database_sqli_outage", "diagnosis")
    assert "Schema rules:" in addendum
    assert "Episode lessons:" in addendum
    assert "query_logs" in addendum


def test_correction_memory_prefers_same_scenario_then_falls_back() -> None:
    memory = CorrectionMemory()
    memory.schema_examples["database_sqli_outage"] = []

    memory.schema_examples["cache_abuse_broken_access_control"] = [
        MemoryExample(
            scenario_id="cache_abuse_broken_access_control",
            stage="diagnosis",
            prompt_text="cache prompt",
            raw_output="bad",
            corrected_action={"action_type": "query_metrics", "service": "cache", "metric": "cpu"},
            failure_type="invalid_json",
            bucket="schema",
            action_family="investigate",
            mistake="Bad JSON.",
            correction="Return JSON only.",
        )
    ]
    addendum = memory.build_prompt_addendum("database_sqli_outage", "diagnosis")
    assert "query_metrics" in addendum


def test_replay_dataset_includes_teacher_and_student_actions(tmp_path: Path) -> None:
    runner = EpisodeRunner(
        backend=SequenceBackend(_action_jsons("database_sqli_outage")),
        parser=StrictActionParser(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    record = runner.run("database_sqli_outage", mode="strict")
    store = TrajectoryStore(tmp_path / "episodes.jsonl")
    store.append_episode(record)

    replay_rows = build_replay_records(tmp_path / "episodes.jsonl")
    assert replay_rows
    assert replay_rows[0].target_action is not None
    assert replay_rows[0].student_action is not None
    assert replay_rows[0].parse_status in {"ok", "repaired"}


def test_eval_summary_reports_success_and_schema_failure_rates() -> None:
    summary = summarize(
        [
            EvalScenarioResult(
                model_name="m1",
                scenario_id="database_sqli_outage",
                success=True,
                final_score=0.9,
                failure_reason=None,
                schema_failure=False,
                elapsed_s=1.0,
            ),
            EvalScenarioResult(
                model_name="m1",
                scenario_id="cache_abuse_broken_access_control",
                success=False,
                final_score=0.1,
                failure_reason="parse_failure:invalid_json",
                schema_failure=True,
                elapsed_s=1.0,
            ),
        ],
        mode="strict",
    )
    assert summary.success_rate == 0.5
    assert summary.avg_score == 0.5
    assert summary.schema_failure_rate == 0.5
    assert summary.by_model["m1"]["success_rate"] == 0.5
