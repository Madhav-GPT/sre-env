"""Policy adapter artifacts produced by the external trainer wrapper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PolicyAdapter:
    """Serializable prompt-side policy adapter for later phases."""

    base_model: str
    phase_name: str
    schema_examples: list[dict] = field(default_factory=list)
    next_action_examples: list[dict] = field(default_factory=list)
    recovery_examples: list[dict] = field(default_factory=list)

    def build_prompt_addendum(self, scenario_id: str, stage: str) -> str:
        """Build an addendum for the given scenario and stage."""
        lines: list[str] = []
        schema = [
            item for item in self.schema_examples
            if item.get("scenario_id") == scenario_id
        ][:2]
        next_action = [
            item for item in self.next_action_examples
            if item.get("scenario_id") == scenario_id
        ][:2]
        recovery = [
            item for item in self.recovery_examples
            if item.get("scenario_id") == scenario_id
        ][:2]

        if schema:
            lines.append("Adapter schema reminders:")
            for item in schema:
                lines.append(
                    f"- invalid -> valid: {json.dumps(item['target_action'], separators=(',', ':'))}"
                )
        if next_action:
            lines.append("Adapter next-action reminders:")
            for item in next_action:
                lines.append(
                    f"- stage {item.get('workflow_stage', stage)} -> {json.dumps(item['target_action'], separators=(',', ':'))}"
                )
        if recovery:
            lines.append("Adapter recovery reminders:")
            for item in recovery:
                lines.append(
                    f"- recover with {json.dumps(item['target_action'], separators=(',', ':'))}"
                )
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Write adapter JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PolicyAdapter":
        """Load adapter JSON from disk."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))
