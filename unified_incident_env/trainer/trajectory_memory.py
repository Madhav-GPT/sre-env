"""Compact policy-card memory for in-session improvement without weight updates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .types import EpisodeRecord, FailureAnalysisReport

SCHEMA_RULES = [
    "Return JSON only.",
    "Use action_type.",
    "Use only allowed actions.",
    "Do not add explanation text.",
]
MAX_POLICY_WORDS = 300


@dataclass
class MemoryExample:
    """One stored correction example."""

    scenario_id: str
    stage: str
    prompt_text: str
    raw_output: str
    corrected_action: dict
    tags: list[str] = field(default_factory=list)
    bucket: str = "next_action"
    failure_type: str | None = None
    action_family: str | None = None
    mistake: str = ""
    correction: str = ""


@dataclass
class CorrectionMemory:
    """Stores compact policy lessons and examples across episodes."""

    max_examples_per_bucket: int = 6
    schema_examples: dict[str, list[MemoryExample]] = field(default_factory=dict)
    next_action_examples: dict[str, list[MemoryExample]] = field(default_factory=dict)
    recovery_examples: dict[str, list[MemoryExample]] = field(default_factory=dict)

    def add_phase_examples(
        self,
        episodes: list[EpisodeRecord],
        analyses: list[FailureAnalysisReport],
    ) -> None:
        analysis_map = {
            analysis.episode_ids[0]: analysis for analysis in analyses if analysis.episode_ids
        }
        for episode in episodes:
            analysis = analysis_map.get(episode.episode_id or 0)
            if analysis is not None:
                self.add_episode_examples(episode, analysis)

    def add_episode_examples(
        self,
        episode: EpisodeRecord,
        analysis: FailureAnalysisReport,
    ) -> None:
        """Update memory immediately after one episode."""
        entries_by_step: dict[int, list] = {}
        for entry in analysis.entries:
            if entry.step_index is None:
                continue
            entries_by_step.setdefault(entry.step_index, []).append(entry)

        for step in episode.step_records:
            if step.teacher_action is None:
                continue
            step_entries = entries_by_step.get(step.step_index, [])
            if not step_entries and step.parse_status not in {
                "invalid_json",
                "invalid_action",
                "repaired",
                "teacher_override",
            }:
                continue

            primary = step_entries[0] if step_entries else None
            failure_type = primary.failure_type if primary else _schema_failure_type(step.parse_status)
            bucket = primary.bucket if primary else "schema"
            mistake = _mistake_text(step, primary)
            correction = _correction_text(step)
            example = MemoryExample(
                scenario_id=episode.scenario_id,
                stage=step.workflow_stage,
                prompt_text=step.prompt_text,
                raw_output=step.raw_model_output,
                corrected_action=step.teacher_action,
                tags=list({entry.failure_type for entry in step_entries} or ([failure_type] if failure_type else [])),
                bucket=bucket,
                failure_type=failure_type,
                action_family=_action_family(step.teacher_action),
                mistake=mistake,
                correction=correction,
            )

            if bucket == "schema" or step.parse_status in {
                "invalid_json",
                "invalid_action",
                "repaired",
                "teacher_override",
            }:
                self._append(self.schema_examples, episode.scenario_id, example)
            if bucket in {"policy", "reasoning"}:
                self._append(self.next_action_examples, episode.scenario_id, example)
            if bucket == "looping":
                self._append(self.recovery_examples, episode.scenario_id, example)

    def build_prompt_addendum(self, scenario_id: str, stage: str) -> str:
        """Return a compact rolling policy card for the current scenario and stage."""
        schema = self._select_examples(self.schema_examples, scenario_id, stage, limit=1)
        failures = self._select_examples(self.next_action_examples, scenario_id, stage, limit=1)
        recoveries = self._select_examples(self.recovery_examples, scenario_id, stage, limit=1)
        examples = self._select_examples(
            {
                **self.schema_examples,
                **{
                    key: self.next_action_examples.get(key, []) + self.recovery_examples.get(key, [])
                    for key in set(self.next_action_examples) | set(self.recovery_examples)
                },
            },
            scenario_id,
            stage,
            limit=1,
        )

        lines = ["Schema rules:"]
        lines.extend(f"- {rule}" for rule in SCHEMA_RULES)

        lesson_lines: list[str] = []
        for item in schema + failures + recoveries:
            lesson_lines.append(f"- Mistake: {item.mistake}")
            lesson_lines.append(f"- Correction: {item.correction}")

        if lesson_lines:
            lines.append("")
            lines.append("Episode lessons:")
            lines.extend(lesson_lines[:6])

        if failures:
            item = failures[0]
            lines.append("")
            lines.append("Relevant past failure:")
            lines.append(f"- {item.failure_type or 'wrong_action'}: {item.mistake}")

        if recoveries:
            item = recoveries[0]
            lines.append("")
            lines.append("Relevant recovery:")
            lines.append(
                f"- recover with {json.dumps(item.corrected_action, separators=(',', ':'))}"
            )

        if examples:
            item = examples[0]
            lines.append("")
            lines.append("Valid example:")
            lines.append(
                f"- {json.dumps(item.corrected_action, separators=(',', ':'))}"
            )

        rendered = "\n".join(lines)
        return _limit_words(rendered, MAX_POLICY_WORDS)

    def stats(self) -> dict[str, int]:
        """Return compact memory counts."""
        return {
            "schema_examples": sum(len(v) for v in self.schema_examples.values()),
            "next_action_examples": sum(len(v) for v in self.next_action_examples.values()),
            "recovery_examples": sum(len(v) for v in self.recovery_examples.values()),
        }

    def merge(self, other: "CorrectionMemory") -> None:
        """Merge another memory store into this one."""
        for bucket_name in ("schema_examples", "next_action_examples", "recovery_examples"):
            current = getattr(self, bucket_name)
            incoming = getattr(other, bucket_name)
            for key, values in incoming.items():
                current.setdefault(key, [])
                current[key].extend(values)
                current[key] = current[key][-self.max_examples_per_bucket :]

    def save(self, path: Path) -> None:
        """Persist memory to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_examples": self._dump(self.schema_examples),
                    "next_action_examples": self._dump(self.next_action_examples),
                    "recovery_examples": self._dump(self.recovery_examples),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CorrectionMemory":
        """Load memory from disk."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        memory = cls()
        memory.schema_examples = memory._load_bucket(data.get("schema_examples", {}))
        memory.next_action_examples = memory._load_bucket(data.get("next_action_examples", {}))
        memory.recovery_examples = memory._load_bucket(data.get("recovery_examples", {}))
        return memory

    def _append(self, bucket: dict[str, list[MemoryExample]], key: str, example: MemoryExample) -> None:
        bucket.setdefault(key, []).append(example)
        bucket[key] = bucket[key][-self.max_examples_per_bucket :]

    def _select_examples(
        self,
        bucket: dict[str, list[MemoryExample]],
        scenario_id: str,
        stage: str,
        *,
        limit: int,
    ) -> list[MemoryExample]:
        all_examples = [
            item
            for items in bucket.values()
            for item in items
        ]
        ranked = sorted(
            all_examples,
            key=lambda item: self._score(item, scenario_id, stage),
            reverse=True,
        )
        selected: list[MemoryExample] = []
        seen = set()
        for item in ranked:
            key = (
                item.scenario_id,
                item.stage,
                item.failure_type,
                json.dumps(item.corrected_action, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _score(self, item: MemoryExample, scenario_id: str, stage: str) -> int:
        score = 0
        if item.scenario_id == scenario_id:
            score += 100
        if item.stage == stage:
            score += 60
        if item.failure_type:
            score += 10
        if item.action_family:
            score += 5
        return score

    def _dump(self, bucket: dict[str, list[MemoryExample]]) -> dict:
        return {
            key: [
                {
                    "scenario_id": item.scenario_id,
                    "stage": item.stage,
                    "prompt_text": item.prompt_text,
                    "raw_output": item.raw_output,
                    "corrected_action": item.corrected_action,
                    "tags": item.tags,
                    "bucket": item.bucket,
                    "failure_type": item.failure_type,
                    "action_family": item.action_family,
                    "mistake": item.mistake,
                    "correction": item.correction,
                }
                for item in value
            ]
            for key, value in bucket.items()
        }

    def _load_bucket(self, bucket: dict) -> dict[str, list[MemoryExample]]:
        return {
            key: [MemoryExample(**item) for item in value]
            for key, value in bucket.items()
        }


def _schema_failure_type(parse_status: str) -> str:
    if parse_status == "invalid_json":
        return "invalid_json"
    return "invalid_action"


def _action_family(action: dict | None) -> str | None:
    action_type = (action or {}).get("action_type")
    if action_type in {"query_logs", "query_metrics", "query_dependencies"}:
        return "investigate"
    if action_type in {
        "inspect_code",
        "classify_vulnerability",
        "apply_patch",
        "verify_security_fix",
        "submit_security_fix",
    }:
        return "security"
    if action_type in {"restart_service", "rollback_deploy"}:
        return "recovery"
    if action_type == "submit_postmortem":
        return "postmortem"
    return None


def _mistake_text(step, entry) -> str:
    if step.failure_type and step.observation.get("why_failed"):
        return step.observation["why_failed"]
    if entry is not None:
        return entry.detail
    return "The previous action did not follow the stage rules."


def _correction_text(step) -> str:
    teacher_action = step.teacher_action or {}
    action_type = teacher_action.get("action_type")
    if action_type is None:
        return "Return one valid action object for the current stage."
    if action_type in {"query_logs", "query_metrics", "query_dependencies"}:
        return f"If you are still diagnosing, prefer {action_type} instead of recovery actions."
    if action_type in {
        "inspect_code",
        "classify_vulnerability",
        "apply_patch",
        "verify_security_fix",
        "submit_security_fix",
    }:
        return f"If security is not completed, prefer {action_type} before infrastructure recovery."
    if action_type in {"restart_service", "rollback_deploy"}:
        return f"After security completion, use {action_type} for recovery."
    return f"The next valid move is {action_type}."


def _limit_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).strip() + " ..."
