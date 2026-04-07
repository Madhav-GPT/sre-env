"""Tests for the ten-episode session loop."""

from __future__ import annotations

import json
from pathlib import Path

from unified_incident_env.scripts.baseline_agent import plan_for_scenario
from unified_incident_env.server.environment import UnifiedIncidentEnvironment
from unified_incident_env.trainer.action_adapter import StrictActionParser
from unified_incident_env.trainer.analyze_failures import analyze_episode
from unified_incident_env.trainer.build_datasets import (
    build_next_action_records,
    build_recovery_records,
    build_schema_repair_records,
)
from unified_incident_env.trainer.collect_trajectory import collect_episode
from unified_incident_env.trainer.run_episode import EpisodeRunner
from unified_incident_env.trainer.run_session import run_session
from unified_incident_env.trainer.session_config import default_phases, default_scenario_schedule, make_session_config
from unified_incident_env.trainer.types import ModelRequest, ModelResponse, UpdateRequest
from unified_incident_env.trainer.update_model import (
    ExternalCommandUpdater,
    NoOpUpdater,
    OpenAIFineTuneUpdater,
    build_updater,
)
from unified_incident_env.trainer.train_external import build_policy_adapter


class SequenceBackend:
    """Backend that returns pre-seeded outputs in order."""

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


def test_default_session_schedule_and_phases() -> None:
    schedule = default_scenario_schedule()
    assert len(schedule) == 10
    assert schedule[:4] == [
        "database_sqli_outage",
        "cache_abuse_broken_access_control",
        "worker_bad_deploy_command_injection",
        "database_sqli_outage",
    ]

    phases = default_phases()
    assert [phase.episode_ids for phase in phases] == [
        [1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10],
    ]
    assert [phase.update_index for phase in phases] == [1, 2, 3, None]


def test_session_config_modes_default_correctly() -> None:
    competition = make_session_config(
        model_name="stub-model",
        output_root="/tmp/trainer-tests",
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
    )
    assert competition.runtime_mode == "competition"
    assert competition.log_rendered_prompts is False
    assert competition.updater_backend == "external_command"

    research = make_session_config(
        model_name="stub-model",
        output_root="/tmp/trainer-tests",
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        runtime_mode="research",
        updater_backend="openai_finetune",
    )
    assert research.runtime_mode == "research"
    assert research.log_rendered_prompts is True


def test_collect_episode_populates_summary_and_analysis() -> None:
    runner = EpisodeRunner(
        backend=SequenceBackend(_action_jsons("database_sqli_outage")),
        parser=StrictActionParser(),
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    record, summary, analysis = collect_episode(
        runner=runner,
        scenario_id="database_sqli_outage",
        episode_id=1,
        mode="strict",
        model_version="stub-model",
    )
    assert record.success is True
    assert summary.success is True
    assert summary.schema_failures == 0
    assert analysis.summary["schema"] == 0
    assert record.episode_id == 1
    assert record.step_records[0].step_index == 1


def test_analyze_failures_buckets_schema_policy_and_looping() -> None:
    strict_parser = StrictActionParser()
    bad_runner = EpisodeRunner(
        backend=SequenceBackend(["- inspect_code"]),
        parser=strict_parser,
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    bad_record = bad_runner.run(
        "database_sqli_outage",
        mode="strict",
        episode_id=1,
        model_version="stub-model",
    )
    bad_analysis = analyze_episode(bad_record)
    assert "invalid_json" in bad_analysis.schema_failures

    wrong_outputs = [
        json.dumps({"action_type": "restart_service", "service": "database"})
        for _ in range(10)
    ]
    wrong_runner = EpisodeRunner(
        backend=SequenceBackend(wrong_outputs),
        parser=strict_parser,
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    wrong_record = wrong_runner.run(
        "database_sqli_outage",
        mode="strict",
        episode_id=2,
        model_version="stub-model",
    )
    wrong_analysis = analyze_episode(wrong_record)
    assert "infra_before_security" in wrong_analysis.policy_failures
    assert "repeated_same_action" in wrong_analysis.looping_failures


def test_dataset_builders_emit_schema_next_action_and_recovery_rows() -> None:
    strict_parser = StrictActionParser()
    good_runner = EpisodeRunner(
        backend=SequenceBackend(_action_jsons("database_sqli_outage")),
        parser=strict_parser,
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    good_record = good_runner.run(
        "database_sqli_outage",
        mode="strict",
        episode_id=1,
        model_version="stub-model",
    )
    good_analysis = analyze_episode(good_record)

    bad_runner = EpisodeRunner(
        backend=SequenceBackend(["- inspect_code"]),
        parser=strict_parser,
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    bad_record = bad_runner.run(
        "database_sqli_outage",
        mode="strict",
        episode_id=2,
        model_version="stub-model",
    )
    bad_analysis = analyze_episode(bad_record)

    wrong_runner = EpisodeRunner(
        backend=SequenceBackend(
            [json.dumps({"action_type": "restart_service", "service": "database"}) for _ in range(10)]
        ),
        parser=strict_parser,
        model_name="stub-model",
        env_factory=UnifiedIncidentEnvironment,
    )
    wrong_record = wrong_runner.run(
        "database_sqli_outage",
        mode="strict",
        episode_id=3,
        model_version="stub-model",
    )
    wrong_analysis = analyze_episode(wrong_record)

    schema_rows = build_schema_repair_records([bad_record], [bad_analysis])
    next_rows = build_next_action_records([wrong_record], [wrong_analysis])
    recovery_rows = build_recovery_records([wrong_record], [wrong_analysis])

    assert schema_rows
    assert schema_rows[0].source == "schema_repair"
    assert next_rows
    assert next_rows[0].source == "next_action"
    assert recovery_rows
    assert recovery_rows[0].source == "recovery"
    assert good_analysis.summary["schema"] == 0


def test_updaters_write_manifests(tmp_path: Path) -> None:
    request = UpdateRequest(
        update_index=1,
        phase_name="probe",
        episodes_used=[1, 2],
        datasets_used=["schema_repair.jsonl"],
        model_before="stub-model",
        output_dir=str(tmp_path / "checkpoints"),
    )

    noop = NoOpUpdater().update(request)
    assert noop.status == "noop"
    assert Path(noop.artifact_paths[0]).exists()
    assert Path(noop.artifact_paths[1]).exists()

    command = (
        "python -c \"from pathlib import Path; "
        f"Path(r'{tmp_path / 'checkpoints_ext' / 'marker.txt'}').parent.mkdir(parents=True, exist_ok=True); "
        f"Path(r'{tmp_path / 'checkpoints_ext' / 'marker.txt'}').write_text('ok')\""
    )
    ext_request = request.model_copy(
        update={
            "update_index": 2,
            "output_dir": str(tmp_path / "checkpoints_ext"),
        }
    )
    external = ExternalCommandUpdater(command).update(ext_request)
    assert external.status == "completed"
    assert Path(tmp_path / "checkpoints_ext" / "marker.txt").exists()


def test_build_policy_adapter_reads_phase_datasets(tmp_path: Path) -> None:
    update_dir = tmp_path / "update_01"
    update_dir.mkdir(parents=True, exist_ok=True)
    schema_row = {
        "scenario_id": "database_sqli_outage",
        "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "prompt"}],
        "target_action": {"action_type": "query_logs", "service": "database"},
        "metadata": {"workflow_stage": "diagnosis"},
    }
    (update_dir / "schema_repair.jsonl").write_text(json.dumps(schema_row) + "\n", encoding="utf-8")
    (update_dir / "next_action.jsonl").write_text(json.dumps(schema_row) + "\n", encoding="utf-8")
    (update_dir / "recovery.jsonl").write_text(json.dumps(schema_row) + "\n", encoding="utf-8")
    request = UpdateRequest(
        update_index=1,
        phase_name="probe",
        episodes_used=[1, 2],
        datasets_used=[],
        model_before="qwen2.5:3b",
        output_dir=str(update_dir),
    )
    adapter = build_policy_adapter(request)
    assert adapter.base_model == "qwen2.5:3b"
    assert len(adapter.schema_examples) == 1
    assert len(adapter.next_action_examples) == 1
    assert len(adapter.recovery_examples) == 0


def test_build_updater_rejects_openai_finetune_in_competition_mode() -> None:
    try:
        build_updater(
            "openai_finetune",
            runtime_mode="competition",
            openai_base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
    except ValueError as exc:
        assert "research mode" in str(exc)
    else:
        raise AssertionError("expected competition-mode rejection")


def test_openai_finetune_updater_is_mockable_in_research_mode(tmp_path: Path) -> None:
    class FakeFiles:
        def create(self, file, purpose):
            assert purpose == "fine-tune"
            return type("FileObj", (), {"id": "file-123"})()

    class FakeJobs:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, training_file, model, suffix=None):
            assert training_file == "file-123"
            assert model == "base-model"
            assert suffix == "probe-01"
            return type("JobObj", (), {"id": "job-123", "status": "queued"})()

        def retrieve(self, job_id):
            assert job_id == "job-123"
            self.calls += 1
            return type(
                "JobObj",
                (),
                {"id": "job-123", "status": "succeeded", "fine_tuned_model": "ft:model-123"},
            )()

    class FakeFineTuning:
        def __init__(self) -> None:
            self.jobs = FakeJobs()

    class FakeClient:
        def __init__(self) -> None:
            self.files = FakeFiles()
            self.fine_tuning = FakeFineTuning()

    training_file = tmp_path / "train.jsonl"
    training_file.write_text('{"messages":[],"completion":"{}"}\n', encoding="utf-8")
    updater = OpenAIFineTuneUpdater(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        client=FakeClient(),
        poll_interval_s=0.0,
        timeout_s=1.0,
    )
    request = UpdateRequest(
        update_index=1,
        phase_name="probe",
        episodes_used=[1, 2],
        datasets_used=[str(training_file)],
        model_before="base-model",
        output_dir=str(tmp_path / "openai_update"),
        runtime_mode="research",
        training_file=str(training_file),
        suffix="probe-01",
    )
    result = updater.update(request)
    assert result.status == "completed"
    assert result.model_after == "ft:model-123"
    assert Path(tmp_path / "openai_update" / "update_01_result.json").exists()


def test_run_session_executes_ten_episodes_and_three_updates(tmp_path: Path) -> None:
    config = make_session_config(
        model_name="stub-model",
        output_root=tmp_path,
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        updater_backend="noop",
    )
    outputs = []
    for scenario_id in config.scenario_schedule:
        outputs.extend(_action_jsons(scenario_id))

    report = run_session(
        config,
        backend=SequenceBackend(outputs),
        updater=NoOpUpdater(),
    )

    assert len(report.episode_summaries) == 10
    assert [update.update_index for update in report.updates] == [1, 2, 3]
    assert [phase.episode_ids for phase in report.phase_reports] == [
        [1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10],
    ]

    output_dir = Path(report.output_dir)
    assert (output_dir / "episodes.jsonl").exists()
    assert (output_dir / "summaries.json").exists()
    assert (output_dir / "phase_delta.json").exists()
    assert (output_dir / "final_report.json").exists()
    assert (output_dir / "baseline_teacher_dataset.jsonl").exists()
    assert (output_dir / "schema_repair.jsonl").exists()
    assert (output_dir / "next_action.jsonl").exists()
    assert (output_dir / "recovery.jsonl").exists()
    assert (output_dir / "sft_dataset.jsonl").exists()
    assert (output_dir / "checkpoints" / "update_01" / "update_01_result.json").exists()


def test_run_session_uses_policy_adapter_model_version_on_successful_update(tmp_path: Path) -> None:
    class SuccessfulUpdater:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, request: UpdateRequest):
            self.calls += 1
            adapter_path = Path(request.output_dir) / "policy_adapter.json"
            adapter_path.write_text(
                json.dumps(
                    {
                        "base_model": request.model_before,
                        "phase_name": request.phase_name,
                        "schema_examples": [],
                        "next_action_examples": [],
                        "recovery_examples": [],
                    }
                ),
                encoding="utf-8",
            )
            return {
                "update_index": request.update_index,
                "phase_name": request.phase_name,
                "updater_backend": "external_command",
                "model_before": request.model_before,
                "model_after": f"policy_adapter::{request.model_before}::{adapter_path}",
                "status": "completed",
                "episodes_used": request.episodes_used,
                "datasets_used": request.datasets_used,
                "artifact_paths": [str(adapter_path)],
                "notes": "ok",
            }

    config = make_session_config(
        model_name="stub-model",
        output_root=tmp_path,
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        updater_backend="external_command",
    )
    outputs = []
    for scenario_id in config.scenario_schedule:
        outputs.extend(_action_jsons(scenario_id))

    report = run_session(
        config,
        backend=SequenceBackend(outputs),
        updater=SuccessfulUpdater(),
    )
    later_versions = {
        summary.episode_id: summary.model_version for summary in report.episode_summaries
    }
    assert later_versions[1] == "stub-model"
    assert later_versions[3].startswith("policy_adapter::")
    assert later_versions[6].startswith("policy_adapter::")


def test_run_session_falls_back_to_previous_model_on_failed_update(tmp_path: Path) -> None:
    class FailingUpdater:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, request: UpdateRequest):
            self.calls += 1
            return type(
                "Result",
                (),
                {
                    "update_index": request.update_index,
                    "phase_name": request.phase_name,
                    "updater_backend": "external_command",
                    "model_before": request.model_before,
                    "model_after": "should-not-be-used",
                    "status": "failed",
                    "episodes_used": request.episodes_used,
                    "datasets_used": request.datasets_used,
                    "artifact_paths": [],
                    "notes": "intentional failure",
                },
            )()

    config = make_session_config(
        model_name="stub-model",
        output_root=tmp_path,
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        runtime_mode="competition",
        updater_backend="noop",
    )
    outputs = []
    for scenario_id in config.scenario_schedule:
        outputs.extend(_action_jsons(scenario_id))

    report = run_session(
        config,
        backend=SequenceBackend(outputs),
        updater=FailingUpdater(),
    )
    final_versions = {
        summary.episode_id: summary.model_version for summary in report.episode_summaries
    }
    assert final_versions[1] == "stub-model"
    assert final_versions[10] == "stub-model"


def test_competition_mode_redacts_prompts_in_written_episodes(tmp_path: Path) -> None:
    config = make_session_config(
        model_name="stub-model",
        output_root=tmp_path,
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        runtime_mode="competition",
        log_rendered_prompts=False,
        updater_backend="noop",
    )
    outputs = []
    for scenario_id in config.scenario_schedule:
        outputs.extend(_action_jsons(scenario_id))

    report = run_session(
        config,
        backend=SequenceBackend(outputs),
        updater=NoOpUpdater(),
    )
    output_dir = Path(report.output_dir)
    first_line = (output_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    stored = json.loads(first_line)
    assert stored["step_records"][0]["prompt_text"] == ""


def test_research_mode_keeps_prompts_in_written_episodes(tmp_path: Path) -> None:
    config = make_session_config(
        model_name="stub-model",
        output_root=tmp_path,
        base_url=None,
        api_base_url="http://stub.local/v1",
        api_key="local",
        runtime_mode="research",
        log_rendered_prompts=True,
        updater_backend="noop",
    )
    outputs = []
    for scenario_id in config.scenario_schedule:
        outputs.extend(_action_jsons(scenario_id))

    report = run_session(
        config,
        backend=SequenceBackend(outputs),
        updater=NoOpUpdater(),
    )
    output_dir = Path(report.output_dir)
    first_line = (output_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    stored = json.loads(first_line)
    assert stored["step_records"][0]["prompt_text"]
