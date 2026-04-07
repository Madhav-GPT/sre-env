"""Build supervised JSONL datasets from baseline and replay trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..scripts.baseline_agent import plan_for_scenario
from ..server.challenge import SCENARIOS
from ..server.environment import UnifiedIncidentEnvironment
from .prompts import TRAINING_SYSTEM_PROMPT
from .trajectory_store import TrajectoryStore
from .types import SFTRecord


def build_baseline_records() -> list[SFTRecord]:
    rows: list[SFTRecord] = []
    for scenario_id in SCENARIOS:
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=scenario_id)
        for step_index, action in enumerate(plan_for_scenario(scenario_id), start=1):
            rows.append(
                SFTRecord(
                    source="baseline",
                    scenario_id=scenario_id,
                    tick=obs.tick_count,
                    messages=[
                        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                        {"role": "user", "content": obs.prompt_text},
                    ],
                    target_action=action.model_dump(exclude_none=True),
                    tags=["teacher", f"step_{step_index}"],
                )
            )
            obs = env.step(action)
    return rows


def build_replay_records(episodes_path: Path) -> list[SFTRecord]:
    rows: list[SFTRecord] = []
    for episode in TrajectoryStore(episodes_path).load_episodes():
        for step in episode.step_records:
            if step.teacher_action is None:
                continue
            tags = [episode.mode, step.parse_status]
            if step.failure_reason:
                tags.append("failure")
            rows.append(
                SFTRecord(
                    source="replay",
                    scenario_id=episode.scenario_id,
                    tick=step.tick,
                    messages=[
                        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                        {"role": "user", "content": step.prompt_text},
                    ],
                    target_action=step.teacher_action,
                    student_action=step.cleaned_action,
                    parse_status=step.parse_status,
                    tags=tags,
                )
            )
    return rows


def write_jsonl(records: list[SFTRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["baseline", "replay", "combined"],
        default="combined",
    )
    parser.add_argument(
        "--episodes",
        default="outputs/trainer/episodes.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    records: list[SFTRecord] = []
    if args.source in {"baseline", "combined"}:
        records.extend(build_baseline_records())
    if args.source in {"replay", "combined"}:
        records.extend(build_replay_records(Path(args.episodes)))
    write_jsonl(records, Path(args.output))
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
