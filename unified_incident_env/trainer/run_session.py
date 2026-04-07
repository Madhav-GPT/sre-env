"""Ten-episode session loop with failure analysis and pluggable updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action_adapter import LenientActionAdapter, StrictActionParser
from .backend import OpenAICompatibleBackend
from .build_datasets import (
    build_baseline_records,
    build_next_action_records,
    build_recovery_records,
    build_schema_repair_records,
    combine_sft_records,
    write_jsonl,
)
from .collect_trajectory import collect_episode
from .reporting import build_phase_deltas, write_session_outputs
from .run_episode import EpisodeRunner
from .session_config import make_session_config_from_env
from .policy_adapter import PolicyAdapter
from .trajectory_memory import CorrectionMemory
from .types import SessionPhaseReport, SessionReport, UpdateRequest, UpdateResult
from .update_model import build_updater


def _extract_policy_adapter_path(model_after: str) -> str | None:
    prefix = "policy_adapter::"
    if not model_after.startswith(prefix):
        return None
    parts = model_after.split("::", 2)
    if len(parts) != 3:
        return None
    return parts[2]


def _base_model_from_version(model_version: str) -> str:
    prefix = "policy_adapter::"
    if model_version.startswith(prefix):
        parts = model_version.split("::", 2)
        if len(parts) == 3:
            return parts[1]
    return model_version


def _memory_from_policy_adapter(path: Path) -> CorrectionMemory:
    adapter = PolicyAdapter.load(path)
    memory = CorrectionMemory()
    from .trajectory_memory import MemoryExample

    for row in adapter.schema_examples:
        memory.schema_examples.setdefault(row["scenario_id"], []).append(
            MemoryExample(
                scenario_id=row["scenario_id"],
                stage=row.get("metadata", {}).get("workflow_stage", "diagnosis"),
                prompt_text=row["messages"][-1]["content"],
                raw_output=json.dumps(row.get("student_action", {}) or {}),
                corrected_action=row["target_action"],
                tags=row.get("tags", []),
                failure_type=(row.get("tags") or [None])[0],
                action_family=None,
                mistake=row.get("metadata", {}).get("failure_reason", ""),
                correction=f"Use {row['target_action'].get('action_type', 'the valid action')} instead.",
            )
        )
    for row in adapter.next_action_examples:
        memory.next_action_examples.setdefault(row["scenario_id"], []).append(
            MemoryExample(
                scenario_id=row["scenario_id"],
                stage=row.get("metadata", {}).get("workflow_stage", "diagnosis"),
                prompt_text=row["messages"][-1]["content"],
                raw_output=json.dumps(row.get("student_action", {}) or {}),
                corrected_action=row["target_action"],
                tags=row.get("tags", []),
                failure_type=(row.get("tags") or [None])[0],
                action_family=None,
                mistake=row.get("metadata", {}).get("failure_reason", ""),
                correction=f"Use {row['target_action'].get('action_type', 'the valid action')} instead.",
            )
        )
    for row in adapter.recovery_examples:
        memory.recovery_examples.setdefault(row["scenario_id"], []).append(
            MemoryExample(
                scenario_id=row["scenario_id"],
                stage=row.get("metadata", {}).get("workflow_stage", "diagnosis"),
                prompt_text=row["messages"][-1]["content"],
                raw_output=json.dumps(row.get("student_action", {}) or {}),
                corrected_action=row["target_action"],
                tags=row.get("tags", []),
                failure_type=(row.get("tags") or [None])[0],
                action_family=None,
                mistake=row.get("metadata", {}).get("failure_reason", ""),
                correction=f"Recover with {row['target_action'].get('action_type', 'the valid action')}.",
            )
        )
    return memory


def run_session(config, *, backend=None, runner_factory=EpisodeRunner, updater=None):
    output_dir = Path(config.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        backend = OpenAICompatibleBackend(
            base_url=config.api_base_url,
            api_key=config.api_key,
        )
    if updater is None:
        updater = build_updater(
            config.updater_backend,
            runtime_mode=config.runtime_mode,
            command_template=config.updater_command_template,
            openai_base_url=config.api_base_url,
            api_key=config.api_key,
        )
    phase_reports: list[SessionPhaseReport] = []

    baseline_records = build_baseline_records()
    write_jsonl(baseline_records, output_dir / "baseline_teacher_dataset.jsonl")
    memory_path = output_dir / "correction_memory.json"
    correction_memory = CorrectionMemory.load(memory_path)

    all_records = []
    all_summaries = []
    all_updates = []
    all_schema_records = []
    all_next_action_records = []
    all_recovery_records = []

    current_model_version = config.initial_model_version

    for phase in config.phases:
        mode = (
            config.final_eval_mode
            if phase.phase_name == "final_evaluation"
            else config.collection_mode
        )
        parser_impl = StrictActionParser() if mode == "strict" else LenientActionAdapter()
        runner = runner_factory(
            backend=backend,
            parser=parser_impl,
            model_name=config.model_name,
            base_url=config.base_url,
            correction_memory=correction_memory,
        )
        phase_records = []
        phase_summaries = []
        phase_analyses = []

        for episode_id in phase.episode_ids:
            scenario_id = config.scenario_schedule[episode_id - 1]
            record, summary, analysis = collect_episode(
                runner=runner,
                scenario_id=scenario_id,
                episode_id=episode_id,
                mode=mode,
                model_version=current_model_version,
            )
            phase_records.append(record)
            phase_summaries.append(summary)
            phase_analyses.append(analysis)
            all_records.append(record)
            all_summaries.append(summary)
            correction_memory.add_episode_examples(record, analysis)
            correction_memory.save(memory_path)

        update_ids = []
        if phase.update_after and phase.update_index is not None:
            update_dir = checkpoints_dir / f"update_{phase.update_index:02d}"
            schema_records = build_schema_repair_records(phase_records, phase_analyses)
            next_action_records = build_next_action_records(phase_records, phase_analyses)
            recovery_records = build_recovery_records(phase_records, phase_analyses)
            combined_records = combine_sft_records(
                baseline_records=[],
                schema_records=schema_records,
                next_action_records=next_action_records,
                recovery_records=recovery_records,
            )
            write_jsonl(schema_records, update_dir / "schema_repair.jsonl")
            write_jsonl(next_action_records, update_dir / "next_action.jsonl")
            write_jsonl(recovery_records, update_dir / "recovery.jsonl")
            write_jsonl(combined_records, update_dir / "sft_dataset.jsonl")

            all_schema_records.extend(schema_records)
            all_next_action_records.extend(next_action_records)
            all_recovery_records.extend(recovery_records)

            update_request = UpdateRequest(
                update_index=phase.update_index,
                phase_name=phase.phase_name,
                episodes_used=phase.episode_ids,
                datasets_used=[
                    str(update_dir / "schema_repair.jsonl"),
                    str(update_dir / "next_action.jsonl"),
                    str(update_dir / "recovery.jsonl"),
                    str(update_dir / "sft_dataset.jsonl"),
                ],
                model_before=_base_model_from_version(current_model_version),
                output_dir=str(update_dir.resolve()),
                runtime_mode=config.runtime_mode,
                command_template=config.updater_command_template,
                training_file=str((update_dir / "sft_dataset.jsonl").resolve()),
                suffix=f"{phase.phase_name[:10]}-{phase.update_index:02d}",
            )
            update_result = updater.update(update_request)
            if not isinstance(update_result, UpdateResult):
                update_result = UpdateResult.model_validate(
                    update_result,
                    from_attributes=True,
                )
            all_updates.append(update_result)
            if update_result.status == "completed":
                current_model_version = update_result.model_after
                adapter_path = _extract_policy_adapter_path(update_result.model_after)
                if adapter_path is not None and Path(adapter_path).exists():
                    correction_memory.merge(
                        _memory_from_policy_adapter(Path(adapter_path))
                    )
                    correction_memory.save(memory_path)
            update_ids.append(update_result.update_index)

        phase_report = SessionPhaseReport(
            phase_name=phase.phase_name,
            episode_ids=phase.episode_ids,
            avg_score=round(
                sum(summary.final_score for summary in phase_summaries) / len(phase_summaries),
                4,
            ),
            success_rate=round(
                sum(1 for summary in phase_summaries if summary.success) / len(phase_summaries),
                4,
            ),
            schema_failures=sum(summary.schema_failures for summary in phase_summaries),
            loop_failures=sum(len(summary.looping_failures) for summary in phase_summaries),
            updates_applied=update_ids,
        )
        phase_reports.append(phase_report)

    write_jsonl(all_schema_records, output_dir / "schema_repair.jsonl")
    write_jsonl(all_next_action_records, output_dir / "next_action.jsonl")
    write_jsonl(all_recovery_records, output_dir / "recovery.jsonl")
    write_jsonl(
        combine_sft_records(
            baseline_records=baseline_records,
            schema_records=all_schema_records,
            next_action_records=all_next_action_records,
            recovery_records=all_recovery_records,
        ),
        output_dir / "sft_dataset.jsonl",
    )

    initial_phase = [summary for summary in all_summaries if summary.episode_id in {1, 2}]
    final_phase = [summary for summary in all_summaries if summary.episode_id in {9, 10}]
    improvement_metrics = {
        "score_delta_ep1_2_to_ep9_10": round(
            (
                sum(item.final_score for item in final_phase) / len(final_phase)
                if final_phase
                else 0.0
            )
            - (
                sum(item.final_score for item in initial_phase) / len(initial_phase)
                if initial_phase
                else 0.0
            ),
            4,
        ),
        "schema_failure_delta_ep1_2_to_ep9_10": round(
            (
                sum(item.schema_failures for item in initial_phase) / len(initial_phase)
                if initial_phase
                else 0.0
            )
            - (
                sum(item.schema_failures for item in final_phase) / len(final_phase)
                if final_phase
                else 0.0
            ),
            4,
        ),
    }

    report = SessionReport(
        session_id=config.session_id,
        model_name=config.model_name,
        runtime_mode=config.runtime_mode,
        output_dir=str(output_dir),
        episode_summaries=all_summaries,
        updates=all_updates,
        phase_reports=phase_reports,
        phase_deltas=build_phase_deltas(phase_reports),
        improvement_metrics=improvement_metrics,
        correction_memory_stats=correction_memory.stats(),
    )
    write_session_outputs(
        output_dir=output_dir,
        report=report,
        records=all_records,
        include_prompts=config.log_rendered_prompts,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-root", default="outputs/trainer")
    parser.add_argument("--runtime-mode", choices=["competition", "research"], default="competition")
    parser.add_argument("--collection-mode", choices=["strict", "lenient"], default="lenient")
    parser.add_argument("--final-eval-mode", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--updater-backend", choices=["noop", "external_command", "openai_finetune"], default="external_command")
    parser.add_argument("--updater-command-template", default=None)
    parser.add_argument("--log-rendered-prompts", action="store_true")
    args = parser.parse_args()

    config = make_session_config_from_env(
        model_name=args.model,
        output_root=args.output_root,
        base_url=args.base_url,
        runtime_mode=args.runtime_mode,
        collection_mode=args.collection_mode,
        final_eval_mode=args.final_eval_mode,
        updater_backend=args.updater_backend,
        updater_command_template=args.updater_command_template,
        log_rendered_prompts=args.log_rendered_prompts if args.log_rendered_prompts else None,
    )
    report = run_session(config)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
