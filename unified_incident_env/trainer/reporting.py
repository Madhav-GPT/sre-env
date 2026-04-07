"""Session reporting and artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .types import EpisodeRecord, PhaseDeltaRecord, SessionPhaseReport, SessionReport


def build_phase_deltas(phase_reports: list[SessionPhaseReport]) -> list[PhaseDeltaRecord]:
    """Compute deltas from one phase to the next."""
    deltas: list[PhaseDeltaRecord] = []
    previous = None
    for report in phase_reports:
        if previous is None:
            delta = PhaseDeltaRecord(
                phase_name=report.phase_name,
                score_delta=0.0,
                schema_failure_delta=0.0,
                loop_failure_delta=0.0,
                success_delta=0.0,
            )
        else:
            delta = PhaseDeltaRecord(
                phase_name=report.phase_name,
                score_delta=round(report.avg_score - previous.avg_score, 4),
                schema_failure_delta=round(
                    (report.schema_failures / len(report.episode_ids))
                    - (previous.schema_failures / len(previous.episode_ids)),
                    4,
                ),
                loop_failure_delta=round(
                    _avg_loop_failures(report) - _avg_loop_failures(previous),
                    4,
                ),
                success_delta=round(report.success_rate - previous.success_rate, 4),
            )
        deltas.append(delta)
        previous = report
    return deltas


def _avg_loop_failures(report: SessionPhaseReport) -> float:
    if not report.episode_ids:
        return 0.0
    return float(report.loop_failures) / len(report.episode_ids)


def redact_episode_prompts(
    records: list[EpisodeRecord],
    *,
    include_prompts: bool,
) -> list[EpisodeRecord]:
    """Redact prompt text fields when compact artifacts are desired."""
    if include_prompts:
        return records
    redacted: list[EpisodeRecord] = []
    for record in records:
        step_records = []
        for step in record.step_records:
            observation = dict(step.observation)
            if "prompt_text" in observation:
                observation["prompt_text"] = ""
            step_records.append(
                step.model_copy(
                    update={
                        "prompt_text": "",
                        "next_prompt_text": None,
                        "observation": observation,
                    }
                )
            )
        redacted.append(record.model_copy(update={"step_records": step_records}))
    return redacted


def write_jsonl_episodes(path: Path, records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def build_final_report(report: SessionReport) -> dict:
    """Build the high-level report JSON shown at the end of a session."""
    summaries = report.episode_summaries
    final_report = {
        "session_id": report.session_id,
        "model_name": report.model_name,
        "runtime_mode": report.runtime_mode,
        "initial_avg_score": 0.0,
        "final_avg_score": 0.0,
        "success_count": 0,
        "schema_failure_rate": 0.0,
        "json_valid_rate": 0.0,
        "strict_schema_valid_rate": 0.0,
        "average_repeated_action_count": 0.0,
        "teacher_override_rate": 0.0,
        "repair_retry_success_rate": 0.0,
        "schema_to_policy_progression_rate": 0.0,
        "security_unlock_rate": 0.0,
        "security_completion_rate": 0.0,
        "full_task_completion_rate": 0.0,
        "average_steps_to_termination": 0.0,
        "average_score_per_scenario": {},
        "phase_to_phase_score_delta": {
            delta.phase_name: delta.score_delta for delta in report.phase_deltas
        },
        "improvement_metrics": report.improvement_metrics,
        "correction_memory_stats": report.correction_memory_stats,
    }
    if summaries:
        initial = [item for item in summaries if item.episode_id in {1, 2}]
        final = [item for item in summaries if item.episode_id in {9, 10}]
        final_report["initial_avg_score"] = round(
            sum(item.final_score for item in initial) / len(initial), 4
        ) if initial else 0.0
        final_report["final_avg_score"] = round(
            sum(item.final_score for item in final) / len(final), 4
        ) if final else 0.0
        final_report["success_count"] = sum(1 for item in summaries if item.success)
        final_report["schema_failure_rate"] = round(
            sum(item.schema_failures for item in summaries) / len(summaries),
            4,
        )
        final_report["json_valid_rate"] = round(
            sum(1 for item in summaries if item.schema_failures == 0) / len(summaries),
            4,
        )
        final_report["strict_schema_valid_rate"] = round(
            sum(
                1
                for item in summaries
                if item.mode == "strict" and item.schema_failures == 0
            )
            / max(1, sum(1 for item in summaries if item.mode == "strict")),
            4,
        )
        final_report["average_repeated_action_count"] = round(
            sum(len(item.looping_failures) for item in summaries) / len(summaries),
            4,
        )
        final_report["teacher_override_rate"] = round(
            sum(item.teacher_override_count for item in summaries)
            / max(1, sum(item.steps for item in summaries)),
            4,
        )
        retry_steps = sum(item.repair_retry_count for item in summaries)
        repaired_steps = sum(
            1
            for item in summaries
            if item.repair_retry_count > 0 and item.schema_failures == 0
        )
        final_report["repair_retry_success_rate"] = round(
            repaired_steps / max(1, retry_steps),
            4,
        )
        policy_after_schema = sum(
            1
            for item in summaries
            if item.schema_failures == 0 and item.policy_failures
        )
        final_report["schema_to_policy_progression_rate"] = round(
            policy_after_schema / len(summaries),
            4,
        )
        final_report["security_unlock_rate"] = round(
            sum(
                1
                for item in summaries
                if item.stopped_reason not in {"diagnosis", "root_cause_analysis"}
            )
            / len(summaries),
            4,
        )
        final_report["security_completion_rate"] = round(
            sum(
                1
                for item in summaries
                if "infra_before_security" not in item.policy_failures
                and item.stopped_reason not in {"security_subquest"}
            )
            / len(summaries),
            4,
        )
        final_report["full_task_completion_rate"] = round(
            sum(1 for item in summaries if item.success) / len(summaries),
            4,
        )
        final_report["average_steps_to_termination"] = round(
            sum(item.steps for item in summaries) / len(summaries),
            4,
        )
        by_scenario: dict[str, list[float]] = {}
        for item in summaries:
            by_scenario.setdefault(item.scenario_id, []).append(item.final_score)
        final_report["average_score_per_scenario"] = {
            scenario_id: round(sum(scores) / len(scores), 4)
            for scenario_id, scores in by_scenario.items()
        }
    return final_report


def write_session_outputs(
    *,
    output_dir: Path,
    report: SessionReport,
    records: list[EpisodeRecord],
    include_prompts: bool,
) -> None:
    """Write all top-level session artifacts."""
    redacted_records = redact_episode_prompts(records, include_prompts=include_prompts)
    write_jsonl_episodes(output_dir / "episodes.jsonl", redacted_records)
    (output_dir / "summaries.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "phase_delta.json").write_text(
        json.dumps([item.model_dump() for item in report.phase_deltas], indent=2),
        encoding="utf-8",
    )
    (output_dir / "final_report.json").write_text(
        json.dumps(build_final_report(report), indent=2),
        encoding="utf-8",
    )
