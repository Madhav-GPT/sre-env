"""Collection wrapper that turns one episode into trajectory + analysis + summary."""

from __future__ import annotations

from .analyze_failures import analyze_episode
from .types import EpisodeRecord, EpisodeSummaryRecord


def collect_episode(
    *,
    runner,
    scenario_id: str,
    episode_id: int,
    mode: str,
    model_version: str,
) -> tuple[EpisodeRecord, EpisodeSummaryRecord, object]:
    """Run, analyze, and summarize one episode."""
    record = runner.run(
        scenario_id=scenario_id,
        mode=mode,
        episode_id=episode_id,
        model_version=model_version,
    )
    analysis = analyze_episode(record)
    record.schema_failures = analysis.summary.get("schema", 0)
    record.policy_failures = analysis.policy_failures
    record.looping_failures = analysis.looping_failures
    record.reasoning_failures = analysis.reasoning_failures
    summary = EpisodeSummaryRecord(
        episode_id=episode_id,
        run_id=record.run_id,
        scenario_id=record.scenario_id,
        difficulty=record.difficulty,
        model_name=record.model_name,
        model_version=record.model_version,
        mode=record.mode,
        steps=record.steps,
        success=record.success,
        final_score=record.final_score,
        schema_failures=analysis.summary.get("schema", 0),
        json_valid_steps=record.json_valid_steps,
        strict_schema_valid_steps=record.strict_schema_valid_steps,
        teacher_override_count=record.teacher_override_count,
        repair_retry_count=record.repair_retry_count,
        policy_failures=analysis.policy_failures,
        looping_failures=analysis.looping_failures,
        reasoning_failures=analysis.reasoning_failures,
        security_subquest_completed=record.security_subquest_completed,
        postmortem_completed=record.postmortem_completed,
        stopped_reason=record.stopped_reason,
        elapsed_s=record.elapsed_s,
    )
    return record, summary, analysis
