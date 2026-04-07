"""Failure analysis for episode trajectories."""

from __future__ import annotations

from collections import Counter

from .types import EpisodeRecord, FailureAnalysisReport, FailureBucketEntry, StepRecord

_INFRA_ACTIONS = {"restart_service", "rollback_deploy"}


def analyze_episode(record: EpisodeRecord) -> FailureAnalysisReport:
    """Classify one episode into schema, policy, looping, and reasoning buckets."""
    entries: list[FailureBucketEntry] = []

    for step in record.step_records:
        entries.extend(_classify_step(record, step))

    entries.extend(_classify_episode_level(record))

    schema = sorted({entry.failure_type for entry in entries if entry.bucket == "schema"})
    policy = sorted({entry.failure_type for entry in entries if entry.bucket == "policy"})
    looping = sorted({entry.failure_type for entry in entries if entry.bucket == "looping"})
    reasoning = sorted({entry.failure_type for entry in entries if entry.bucket == "reasoning"})
    summary = Counter(entry.bucket for entry in entries)

    return FailureAnalysisReport(
        episode_ids=[record.episode_id or 0],
        scenario_ids=[record.scenario_id],
        entries=entries,
        schema_failures=schema,
        policy_failures=policy,
        looping_failures=looping,
        reasoning_failures=reasoning,
        summary={
            "schema": summary.get("schema", 0),
            "policy": summary.get("policy", 0),
            "looping": summary.get("looping", 0),
            "reasoning": summary.get("reasoning", 0),
        },
    )


def analyze_block(records: list[EpisodeRecord]) -> FailureAnalysisReport:
    """Combine multiple episode analyses into one block report."""
    analyses = [analyze_episode(record) for record in records]
    entries = [entry for analysis in analyses for entry in analysis.entries]
    summary = Counter(entry.bucket for entry in entries)
    return FailureAnalysisReport(
        episode_ids=[record.episode_id or 0 for record in records],
        scenario_ids=[record.scenario_id for record in records],
        entries=entries,
        schema_failures=sorted({entry.failure_type for entry in entries if entry.bucket == "schema"}),
        policy_failures=sorted({entry.failure_type for entry in entries if entry.bucket == "policy"}),
        looping_failures=sorted({entry.failure_type for entry in entries if entry.bucket == "looping"}),
        reasoning_failures=sorted({entry.failure_type for entry in entries if entry.bucket == "reasoning"}),
        summary={
            "schema": summary.get("schema", 0),
            "policy": summary.get("policy", 0),
            "looping": summary.get("looping", 0),
            "reasoning": summary.get("reasoning", 0),
        },
    )


def _classify_step(record: EpisodeRecord, step: StepRecord) -> list[FailureBucketEntry]:
    entries: list[FailureBucketEntry] = []
    if step.parse_status in {"invalid_json", "invalid_action"}:
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="schema",
                failure_type=_schema_failure_type(step),
                detail=step.failure_reason or "schema failure",
            )
        )
        return entries

    student = step.cleaned_action or {}
    teacher = step.teacher_action or {}
    if not teacher or not student or student == teacher:
        return entries

    student_type = student.get("action_type")
    teacher_type = teacher.get("action_type")

    if student_type == "classify_vulnerability":
        failure_type = (
            "wrong_vulnerability"
            if teacher_type == "classify_vulnerability"
            else "fails_to_identify_real_vulnerability"
        )
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="reasoning",
                failure_type=failure_type,
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    if student_type == "apply_patch" and teacher_type == "apply_patch":
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="policy",
                failure_type="wrong_patch",
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    if student_type == "verify_security_fix" and teacher_type != "verify_security_fix":
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="policy",
                failure_type="verify_too_early",
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    if student_type == "submit_security_fix" and teacher_type != "submit_security_fix":
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="policy",
                failure_type="submit_too_early",
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    if student_type in _INFRA_ACTIONS and teacher_type not in _INFRA_ACTIONS:
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="policy",
                failure_type="infra_before_security",
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    if student_type in _INFRA_ACTIONS and teacher_type in _INFRA_ACTIONS:
        failure_type = "wrong_service"
        if student_type == "restart_service":
            failure_type = "wrong_restart"
        elif student_type == "rollback_deploy":
            failure_type = "wrong_rollback"
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=step.step_index,
                bucket="policy",
                failure_type=failure_type,
                detail=f"student={student} teacher={teacher}",
            )
        )
        return entries

    entries.append(
        FailureBucketEntry(
            episode_id=record.episode_id or 0,
            scenario_id=record.scenario_id,
            step_index=step.step_index,
            bucket="policy",
            failure_type="wrong_action_choice",
            detail=f"student={student} teacher={teacher}",
        )
    )
    return entries


def _classify_episode_level(record: EpisodeRecord) -> list[FailureBucketEntry]:
    entries: list[FailureBucketEntry] = []
    previous = None
    repeat_count = 0
    for step in record.step_records:
        current = step.cleaned_action
        if current and current == previous:
            repeat_count += 1
            if repeat_count >= 1:
                entries.append(
                    FailureBucketEntry(
                        episode_id=record.episode_id or 0,
                        scenario_id=record.scenario_id,
                        step_index=step.step_index,
                        bucket="looping",
                        failure_type="repeated_same_action",
                        detail=f"action={current}",
                    )
                )
        else:
            repeat_count = 0
        previous = current

    stopped = record.stopped_reason or ""
    if stopped in {"diagnosis", "root_cause_analysis"}:
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=None,
                bucket="looping",
                failure_type="stuck_in_diagnosis",
                detail=f"stopped_reason={stopped}",
            )
        )
    elif stopped == "security_subquest":
        entries.append(
            FailureBucketEntry(
                episode_id=record.episode_id or 0,
                scenario_id=record.scenario_id,
                step_index=None,
                bucket="looping",
                failure_type="stuck_in_security_subquest",
                detail=f"stopped_reason={stopped}",
            )
        )

    return entries


def _schema_failure_type(step: StepRecord) -> str:
    raw = step.raw_model_output.lower()
    error = (step.failure_reason or "").lower()
    if '"reason"' in raw or '"details"' in raw or "extra_forbidden" in error:
        return "extra_unsupported_fields"
    if '"services"' in raw or '"metrics"' in raw or "field required" in error:
        return "wrong_field_names"
    if "required" in error or "missing" in error:
        return "missing_required_fields"
    if step.parse_status == "invalid_json":
        return "invalid_json"
    return "invalid_action"
