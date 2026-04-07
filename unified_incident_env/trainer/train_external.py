"""Default lightweight external trainer wrapper for session updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy_adapter import PolicyAdapter
from .types import UpdateRequest


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_policy_adapter(request: UpdateRequest) -> PolicyAdapter:
    """Build a lightweight policy adapter from phase datasets."""
    schema_rows = _load_jsonl(Path(request.output_dir) / "schema_repair.jsonl")
    next_action_rows = _load_jsonl(Path(request.output_dir) / "next_action.jsonl")
    recovery_rows = _load_jsonl(Path(request.output_dir) / "recovery.jsonl")

    if request.phase_name == "probe":
        schema_rows = schema_rows[:12]
        next_action_rows = next_action_rows[:4]
        recovery_rows = []
    elif request.phase_name == "first_correction":
        schema_rows = schema_rows[:12]
        next_action_rows = next_action_rows[:12]
        recovery_rows = []
    else:
        schema_rows = schema_rows[:8]
        next_action_rows = next_action_rows[:8]
        recovery_rows = recovery_rows[:12]

    return PolicyAdapter(
        base_model=_base_model_from_version(request.model_before),
        phase_name=request.phase_name,
        schema_examples=schema_rows,
        next_action_examples=next_action_rows,
        recovery_examples=recovery_rows,
    )


def _base_model_from_version(model_version: str) -> str:
    prefix = "policy_adapter::"
    if model_version.startswith(prefix):
        parts = model_version.split("::", 2)
        if len(parts) == 3:
            return parts[1]
    return model_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request = UpdateRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_policy_adapter(request)
    adapter_path = output_dir / "policy_adapter.json"
    adapter.save(adapter_path)
    result = {
        "status": "completed",
        "model_after": f"policy_adapter::{request.model_before}::{adapter_path}",
        "notes": (
            f"built adapter with schema={len(adapter.schema_examples)} "
            f"next_action={len(adapter.next_action_examples)} "
            f"recovery={len(adapter.recovery_examples)}"
        ),
        "artifact_paths": [str(adapter_path)],
    }
    result_path = output_dir / f"update_{request.update_index:02d}_external_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
