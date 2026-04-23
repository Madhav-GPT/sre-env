#!/usr/bin/env python3
"""CLI client for the sre-gym skill.

Usage:
    sre_gym_client.py list
    sre_gym_client.py solve <scenario_id> [--policy baseline]
    sre_gym_client.py interactive <scenario_id>   # stdin: one JSON action per line
    sre_gym_client.py record-runbook <scenario_id> <session.json>

Because OpenEnv's HTTP /reset and /step handlers create a fresh environment per
call, episode state only persists within a single client session. This CLI wraps
one episode inside one Python process so the session is preserved.

SRE_GYM_URL env var overrides the base URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the sibling package importable whether the script is invoked from the
# repo root or from the skill/ directory directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unified_incident_env.client import UnifiedIncidentEnv  # noqa: E402
from unified_incident_env.models import UnifiedIncidentAction, UnifiedIncidentObservation  # noqa: E402
from unified_incident_env.server.challenge import SCENARIOS, list_baselines  # noqa: E402

BASE_URL = os.environ.get("SRE_GYM_URL", "http://127.0.0.1:8000").rstrip("/")
RUNBOOK_DIR = Path(__file__).resolve().parent.parent / "verified-runbooks"
SCORE_THRESHOLD = 0.85


def _clean_action(action: UnifiedIncidentAction) -> dict[str, Any]:
    data = action.model_dump(exclude_none=True)
    if data.get("metadata") == {}:
        data.pop("metadata")
    hypothesis = data.get("hypothesis")
    if isinstance(hypothesis, dict) and hypothesis.get("metadata") == {}:
        hypothesis.pop("metadata", None)
    return data


def _summarize_obs(obs: UnifiedIncidentObservation) -> dict[str, Any]:
    return {
        "tick": obs.tick_count,
        "workflow_stage": obs.workflow_stage,
        "last_action_result": obs.last_action_result,
        "tool_output": obs.tool_output,
        "failure_type": obs.failure_type,
        "why_failed": obs.why_failed,
        "loop_warning": obs.loop_warning,
        "checks": [{"name": c.name, "passed": c.passed} for c in obs.checks],
        "final_score": obs.final_score,
        "incident_resolved": obs.incident_resolved,
    }


def _session_path(scenario_id: str) -> Path:
    return Path(f"/tmp/sre_gym_session.{scenario_id}.json")


def cmd_list() -> None:
    for scenario in SCENARIOS.values():
        print(f"  {scenario['difficulty']:<6} {scenario['id']:<25} {scenario['name']}")


def cmd_solve(scenario_id: str, policy: str = "baseline") -> None:
    """Run an entire episode end-to-end inside one process."""
    if scenario_id not in SCENARIOS:
        print(f"error: unknown scenario {scenario_id!r}", file=sys.stderr)
        sys.exit(2)
    if policy != "baseline":
        print(f"error: unknown policy {policy!r} (only 'baseline' available)", file=sys.stderr)
        sys.exit(2)

    trace: list[dict[str, Any]] = []
    with UnifiedIncidentEnv(base_url=BASE_URL).sync() as env:
        obs = env.reset(scenario_id=scenario_id).observation
        print(f"[reset] scenario={scenario_id} difficulty={obs.difficulty}")
        for step in list_baselines(scenario_id).baselines[0].actions:
            result = env.step(step.action)
            obs = result.observation
            record = {
                "step": obs.tick_count,
                "action": _clean_action(step.action),
                "rationale": step.rationale,
                "reward": result.reward,
                **_summarize_obs(obs),
            }
            trace.append(record)
            action_repr = json.dumps(record["action"], separators=(",", ":"))
            print(f"[step {obs.tick_count}] action={action_repr} reward={result.reward:+.2f} score={obs.final_score:.2f}")
            if result.done:
                break
        final = _summarize_obs(obs)

    _session_path(scenario_id).write_text(
        json.dumps({"scenario_id": scenario_id, "trace": trace, "final": final}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] resolved={final['incident_resolved']} score={final['final_score']:.2f} "
        f"steps={final['tick']} session={_session_path(scenario_id)}"
    )


def cmd_interactive(scenario_id: str) -> None:
    """One JSON action per stdin line. Preserves session for the whole process lifetime."""
    if scenario_id not in SCENARIOS:
        print(f"error: unknown scenario {scenario_id!r}", file=sys.stderr)
        sys.exit(2)

    trace: list[dict[str, Any]] = []
    with UnifiedIncidentEnv(base_url=BASE_URL).sync() as env:
        obs = env.reset(scenario_id=scenario_id).observation
        print(json.dumps({"event": "reset", "scenario_id": scenario_id, "obs": _summarize_obs(obs)}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                action = UnifiedIncidentAction(**data)
            except Exception as exc:
                print(json.dumps({"event": "error", "detail": str(exc)}), flush=True)
                continue
            result = env.step(action)
            obs = result.observation
            record = {"step": obs.tick_count, "action": _clean_action(action), "reward": result.reward, **_summarize_obs(obs)}
            trace.append(record)
            print(json.dumps({"event": "step", **record}), flush=True)
            if result.done:
                print(json.dumps({"event": "done", "final": _summarize_obs(obs)}), flush=True)
                break

    _session_path(scenario_id).write_text(
        json.dumps({"scenario_id": scenario_id, "trace": trace, "final": _summarize_obs(obs)}, indent=2),
        encoding="utf-8",
    )


def cmd_record_runbook(scenario_id: str, session_file: str | None = None) -> None:
    """Append a new runbook entry if the referenced session cleared the threshold."""
    path = Path(session_file) if session_file else _session_path(scenario_id)
    if not path.exists():
        print(f"error: no session file at {path}", file=sys.stderr)
        sys.exit(2)
    session = json.loads(path.read_text(encoding="utf-8"))
    final = session.get("final", {})
    score = float(final.get("final_score", 0.0))

    if not final.get("incident_resolved"):
        print(f"skip: session not resolved (resolved={final.get('incident_resolved')})", file=sys.stderr)
        sys.exit(1)
    if score < SCORE_THRESHOLD:
        print(f"skip: score {score:.2f} below runbook threshold {SCORE_THRESHOLD:.2f}", file=sys.stderr)
        sys.exit(1)

    RUNBOOK_DIR.mkdir(parents=True, exist_ok=True)
    runbook_path = RUNBOOK_DIR / f"{scenario_id}.md"

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    steps = int(final.get("tick", 0))
    checks_passed = [c["name"] for c in final.get("checks", []) if c.get("passed")]
    trace = session.get("trace", [])

    header = (
        f"# verified-runbooks/{scenario_id}.md\n\n"
        "Runbook entries are written by the sre-gym skill after a successful solve "
        f"(incident_resolved=true and final_score > {SCORE_THRESHOLD:.2f}).\n"
        "Each entry is immutable evidence — treat it as ground truth for the winning path.\n\n---\n"
    )
    lines = [f"\n## Run {timestamp} — Score {score:.2f}\n"]
    lines.append(f"- Steps: **{steps}**")
    lines.append(f"- Checks passed: {', '.join(checks_passed) or 'none'}")
    lines.append("")
    lines.append("**Winning path:**")
    for entry in trace:
        act = entry["action"]
        action_type = act.get("action_type")
        extras = ", ".join(
            f"{k}={v if not isinstance(v, dict) else v.get('root_cause', v)}"
            for k, v in act.items()
            if k != "action_type" and v not in (None, {})
        )
        extra_str = f" ({extras})" if extras else ""
        rationale = entry.get("rationale", "").rstrip(".")
        lines.append(f"{entry['step']}. `{action_type}{extra_str}` — {rationale}")
    lines.append("")
    entry_text = "\n".join(lines)

    if not runbook_path.exists():
        runbook_path.write_text(header + entry_text, encoding="utf-8")
    else:
        with runbook_path.open("a", encoding="utf-8") as f:
            f.write(entry_text)
    print(f"recorded runbook entry → {runbook_path} (score {score:.2f}, {steps} steps)")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    cmd, *rest = argv
    if cmd == "list":
        cmd_list()
    elif cmd == "solve":
        if not rest:
            print("error: solve requires <scenario_id>", file=sys.stderr)
            sys.exit(2)
        cmd_solve(rest[0], rest[1] if len(rest) > 1 else "baseline")
    elif cmd == "interactive":
        if not rest:
            print("error: interactive requires <scenario_id>", file=sys.stderr)
            sys.exit(2)
        cmd_interactive(rest[0])
    elif cmd == "record-runbook":
        if not rest:
            print("error: record-runbook requires <scenario_id>", file=sys.stderr)
            sys.exit(2)
        cmd_record_runbook(rest[0], rest[1] if len(rest) > 1 else None)
    else:
        print(f"error: unknown command {cmd!r}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
