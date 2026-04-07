"""Build correction datasets from trajectories and failure analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_sft_dataset import build_baseline_records
from .trajectory_store import TrajectoryStore
from .types import EpisodeRecord, FailureAnalysisReport, SFTRecord


def build_schema_repair_records(
    episodes: list[EpisodeRecord],
    analyses: list[FailureAnalysisReport],
) -> list[SFTRecord]:
    rows: list[SFTRecord] = []
    analysis_by_episode = {
        analysis.episode_ids[0]: analysis for analysis in analyses if analysis.episode_ids
    }
    for episode in episodes:
        analysis = analysis_by_episode.get(episode.episode_id or 0)
        schema_types = set(analysis.schema_failures if analysis else [])
        for step in episode.step_records:
            if step.parse_status not in {"invalid_json", "invalid_action", "repaired", "teacher_override"}:
                continue
            if step.teacher_action is None:
                continue
            rows.append(
                SFTRecord(
                    source="schema_repair",
                    scenario_id=episode.scenario_id,
                    tick=step.tick,
                    messages=[
                        {"role": "system", "content": "Repair the action into strict JSON only."},
                        {
                            "role": "user",
                            "content": (
                                f"{step.prompt_text}\n\n"
                                f"Previous invalid output:\n{step.raw_model_output}"
                            ),
                        },
                    ],
                    target_action=step.teacher_action,
                    student_action=step.cleaned_action,
                    parse_status=step.parse_status,
                    tags=sorted(schema_types) or [step.parse_status],
                    metadata={
                        "episode_id": episode.episode_id,
                        "step_index": step.step_index,
                        "repair_retry_used": step.repair_retry_used,
                        "teacher_override_used": step.teacher_override_used,
                        "normalization_applied": step.normalization_applied,
                        "failure_type": step.observation.get("failure_type"),
                        "why_failed": step.observation.get("why_failed"),
                        "loop_warning": step.observation.get("loop_warning"),
                        "blocked_until_security_complete": step.observation.get("blocked_until_security_complete"),
                        "security_unlock_reason": step.observation.get("security_unlock_reason"),
                        "progress_flags": step.observation.get("progress_flags"),
                    },
                )
            )
    return rows


def build_next_action_records(
    episodes: list[EpisodeRecord],
    analyses: list[FailureAnalysisReport],
) -> list[SFTRecord]:
    rows: list[SFTRecord] = []
    episode_entries = {
        analysis.episode_ids[0]: analysis.entries
        for analysis in analyses
        if analysis.episode_ids
    }
    allowed = {"policy", "reasoning", "looping"}
    for episode in episodes:
        entries = episode_entries.get(episode.episode_id or 0, [])
        step_indices = {
            entry.step_index
            for entry in entries
            if entry.bucket in allowed and entry.step_index is not None
        }
        for step in episode.step_records:
            if step.step_index not in step_indices:
                continue
            if step.teacher_action is None:
                continue
            tags = [
                entry.failure_type
                for entry in entries
                if entry.step_index == step.step_index and entry.bucket in allowed
            ]
            rows.append(
                SFTRecord(
                    source="next_action",
                    scenario_id=episode.scenario_id,
                    tick=step.tick,
                    messages=[
                        {"role": "system", "content": "Choose the best next action as strict JSON only."},
                        {"role": "user", "content": step.prompt_text},
                    ],
                    target_action=step.teacher_action,
                    student_action=step.cleaned_action,
                    parse_status=step.parse_status,
                    tags=sorted(set(tags)) or ["next_action"],
                    metadata={
                        "episode_id": episode.episode_id,
                        "step_index": step.step_index,
                        "workflow_stage": step.workflow_stage,
                        "teacher_override_used": step.teacher_override_used,
                        "failure_type": step.observation.get("failure_type"),
                        "why_failed": step.observation.get("why_failed"),
                        "loop_warning": step.observation.get("loop_warning"),
                        "progress_flags": step.observation.get("progress_flags"),
                    },
                )
            )
    return rows


def build_recovery_records(
    episodes: list[EpisodeRecord],
    analyses: list[FailureAnalysisReport],
) -> list[SFTRecord]:
    rows: list[SFTRecord] = []
    episode_entries = {
        analysis.episode_ids[0]: analysis.entries
        for analysis in analyses
        if analysis.episode_ids
    }
    recovery_failures = {
        "wrong_restart",
        "wrong_rollback",
        "wrong_service",
        "wrong_patch",
        "wrong_vulnerability",
        "verify_too_early",
        "submit_too_early",
        "infra_before_security",
        "repeated_same_action",
    }
    for episode in episodes:
        entries = episode_entries.get(episode.episode_id or 0, [])
        step_indices = {
            entry.step_index
            for entry in entries
            if entry.failure_type in recovery_failures and entry.step_index is not None
        }
        for step in episode.step_records:
            if step.step_index not in step_indices:
                continue
            if step.teacher_action is None or not step.next_prompt_text:
                continue
            tags = [
                entry.failure_type
                for entry in entries
                if entry.step_index == step.step_index
                and entry.failure_type in recovery_failures
            ]
            rows.append(
                SFTRecord(
                    source="recovery",
                    scenario_id=episode.scenario_id,
                    tick=step.tick,
                    messages=[
                        {"role": "system", "content": "Recover from the previous mistake. Return the best next strict JSON action only."},
                        {
                            "role": "user",
                            "content": (
                                f"{step.next_prompt_text}\n\n"
                                f"Previous wrong action: {step.cleaned_action}\n"
                                f"Penalty or result: reward={step.reward}"
                            ),
                        },
                    ],
                    target_action=step.teacher_action,
                    student_action=step.cleaned_action,
                    parse_status=step.parse_status,
                    tags=sorted(set(tags)) or ["recovery"],
                    metadata={
                        "episode_id": episode.episode_id,
                        "step_index": step.step_index,
                        "teacher_override_used": step.teacher_override_used,
                        "failure_type": step.observation.get("failure_type"),
                        "why_failed": step.observation.get("why_failed"),
                        "loop_warning": step.observation.get("loop_warning"),
                        "best_recovery_action_family": step.observation.get("best_recovery_action_family"),
                    },
                )
            )
    return rows


def combine_sft_records(
    *,
    baseline_records: list[SFTRecord],
    schema_records: list[SFTRecord],
    next_action_records: list[SFTRecord],
    recovery_records: list[SFTRecord],
) -> list[SFTRecord]:
    return [
        *baseline_records,
        *schema_records,
        *next_action_records,
        *recovery_records,
    ]


def write_jsonl(records: list[SFTRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def load_episodes(path: Path) -> list[EpisodeRecord]:
    return TrajectoryStore(path).load_episodes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="outputs/trainer/episodes.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes = load_episodes(Path(args.episodes))

    from .analyze_failures import analyze_episode

    analyses = [analyze_episode(episode) for episode in episodes]
    baseline_records = build_baseline_records()
    schema_records = build_schema_repair_records(episodes, analyses)
    next_action_records = build_next_action_records(episodes, analyses)
    recovery_records = build_recovery_records(episodes, analyses)
    combined_records = combine_sft_records(
        baseline_records=baseline_records,
        schema_records=schema_records,
        next_action_records=next_action_records,
        recovery_records=recovery_records,
    )

    write_jsonl(baseline_records, output_dir / "baseline_teacher_dataset.jsonl")
    write_jsonl(schema_records, output_dir / "schema_repair.jsonl")
    write_jsonl(next_action_records, output_dir / "next_action.jsonl")
    write_jsonl(recovery_records, output_dir / "recovery.jsonl")
    write_jsonl(combined_records, output_dir / "sft_dataset.jsonl")
    print(
        f"wrote baseline={len(baseline_records)} schema={len(schema_records)} "
        f"next_action={len(next_action_records)} recovery={len(recovery_records)} "
        f"combined={len(combined_records)} to {output_dir}"
    )


if __name__ == "__main__":
    main()
