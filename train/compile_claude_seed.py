"""Compile raw /tmp/play.py event logs into the canonical training JSONL.

Canonical format matches train/collect_trajectories.py output so downstream
SFT/GRPO pipelines treat these seed trajectories identically to
API-driven ones.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path


def _compile(event_file: Path) -> dict:
    events = [json.loads(line) for line in event_file.read_text().splitlines() if line.strip()]
    reset = next(e for e in events if e["event"] == "reset")
    steps = [e for e in events if e["event"] == "step"]
    evaluation = next((e for e in events if e["event"] == "evaluate"), None)

    scenario_id = reset["scenario_id"]
    prior_obs_str = reset["observation"]
    trajectory = []
    final_obs = None
    for tick, step in enumerate(steps):
        action = {"action_type": step["tool_name"], **step["arguments"]}
        # Hoist hypothesis payload into nested form expected by UnifiedIncidentAction
        if step["tool_name"] == "submit_hypothesis" and "hypothesis" in action:
            action = {"action_type": "submit_hypothesis", "hypothesis": action["hypothesis"]}
        prior_obs = json.loads(prior_obs_str)
        next_obs = json.loads(step["observation"])
        trajectory.append(
            {
                "tick": tick,
                "prompt": prior_obs.get("prompt_text", ""),
                "response_text": json.dumps(action, separators=(",", ":")),
                "action": action,
                "reward": next_obs.get("reward"),
                "tool_output": next_obs.get("tool_output"),
                "failure_type": next_obs.get("failure_type"),
                "workflow_stage": next_obs.get("workflow_stage"),
            }
        )
        prior_obs_str = step["observation"]
        final_obs = next_obs

    final_score = evaluation["evaluation"]["score"] if evaluation else None
    done = bool(final_obs and final_obs.get("done"))
    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario_id,
        "model": "claude-opus-4-7@teacher",
        "final_score": final_score,
        "incident_resolved": done,
        "steps": len(trajectory),
        "elapsed_s": 0.0,
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("train/data"))
    parser.add_argument("--output", type=Path, default=Path("train/data/claude_seed.jsonl"))
    args = parser.parse_args()

    records = []
    for event_file in sorted(args.input_dir.glob("claude_*.jsonl")):
        if event_file.name == args.output.name:
            continue
        # skip already-compiled output files
        first = event_file.read_text().splitlines()[:1]
        if first and '"event"' not in first[0]:
            continue
        record = _compile(event_file)
        records.append(record)
        print(
            f"{event_file.name} -> scenario={record['scenario_id']} "
            f"score={record['final_score']} steps={record['steps']}"
        )

    args.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"wrote {len(records)} episodes -> {args.output}")


if __name__ == "__main__":
    main()
