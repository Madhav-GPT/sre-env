"""Collect Claude Sonnet 4.6 expert trajectories for the 6 missing templates.

Run on a laptop (not inside the JupyterLab Space) so the Anthropic API key
stays out of the Space environment.

Prerequisites:
    1. The local FastAPI env is running:    make dev   (in another shell)
    2. ANTHROPIC_API_KEY is exported in the current shell
    3. pip install anthropic httpx

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python train/collect_sonnet_missing6.py
    git add train/data/sonnet_missing6.jsonl
    git commit -m "data: sonnet trajectories for missing 6 templates"

The output JSONL is consumed by train/build_corpus.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent

MISSING_TEMPLATES = [
    "auth_token_expiry",
    "dep_degradation",
    "memory_leak_oom",
    "migration_lock",
    "network_partition",
    "rate_limit_retry_storm",
]

EXPERT_THRESHOLD = 0.85

SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no code fences.
The 11 actions are:
  query_logs(service)            query_metrics(service, metric)
  query_dependencies(service)    query_deploys(service)
  rollback_deploy(service)       restart_service(service)
  isolate_service(service)       run_check(check_name)
  submit_hypothesis(hypothesis)  escalate
  declare_resolved

Services: api-gateway / cache / database / worker.
metric in {cpu, error_rate, latency}; check_name in {database_recovery, end_to_end}.

A successful episode looks like:
  gather evidence (3-4 queries) -> submit_hypothesis(correct root cause) ->
  rollback_deploy(correct service) -> restart_service(correct service) ->
  run_check(database_recovery) -> run_check(end_to_end) -> declare_resolved.

Wrong rollback / premature restart / premature declare_resolved are penalized.
Repeated identical hypotheses score 0. Watch the loop_warning field.

Respond with one JSON object, e.g.:
{"action_type": "query_deploys", "service": "worker"}
"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.startswith("```")).strip()
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None


def _classify_tier(score: float) -> str:
    if score >= EXPERT_THRESHOLD:
        return "expert"
    if score >= 0.30:
        return "mediocre"
    return "failure"


def collect_episode(
    *,
    env_url: str,
    scenario_id: str,
    model: str,
    client: Any,
    max_steps: int = 16,
) -> dict | None:
    reset = httpx.post(f"{env_url}/reset", json={"scenario_id": scenario_id}, timeout=30.0)
    if reset.status_code != 200:
        print(f"    [reset-fail] {reset.status_code}: {reset.text[:120]}")
        return None
    obs = reset.json().get("observation", reset.json())

    start = time.perf_counter()
    trajectory: list[dict[str, Any]] = []

    for tick in range(max_steps):
        prompt = obs.get("prompt_text") or ""
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(b, "text", "")
                for b in resp.content
                if getattr(b, "type", None) == "text"
            )
        except Exception as exc:
            print(f"    [anthropic-fail] tick {tick}: {exc}")
            return None

        action_dict = _extract_json(text)
        if not action_dict:
            print(f"    [parse-fail] tick {tick}: {text[:100]!r}")
            return None

        step = httpx.post(f"{env_url}/step", json={"action": action_dict}, timeout=30.0)
        if step.status_code != 200:
            print(f"    [step-fail] tick {tick}: {step.status_code} {step.text[:120]}")
            return None
        body = step.json()
        new_obs = body.get("observation", body)
        trajectory.append({
            "tick": int(new_obs.get("tick_count", tick + 1)),
            "prompt": prompt,
            "response_text": json.dumps(action_dict, separators=(",", ":")),
            "action": action_dict,
            "reward": float(body.get("reward", 0.0)),
            "tool_output": new_obs.get("tool_output"),
            "failure_type": new_obs.get("failure_type"),
            "workflow_stage": new_obs.get("workflow_stage", "triage"),
        })
        obs = new_obs
        if body.get("done"):
            break

    elapsed = time.perf_counter() - start
    final_score = float(obs.get("final_score", 0.0))
    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario_id,
        "template_id": scenario_id.split("__")[0],
        "model": model,
        "driver": "anthropic",
        "seed": 0,
        "difficulty": obs.get("difficulty", "medium"),
        "final_score": final_score,
        "incident_resolved": bool(obs.get("incident_resolved")),
        "steps": int(obs.get("tick_count", len(trajectory))),
        "elapsed_s": round(elapsed, 3),
        "score_breakdown": dict(obs.get("score_breakdown") or {}),
        "trajectory": trajectory,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_batch": f"sonnet_missing6_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "quality_tier": _classify_tier(final_score),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--episodes-per-template", type=int, default=5,
                        help="Targets base + __p01..__p04 by default")
    parser.add_argument("--max-attempts-per-scenario", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "train" / "data" / "sonnet_missing6.jsonl",
    )
    args = parser.parse_args(argv)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    try:
        h = httpx.get(f"{args.env_url}/health", timeout=5.0)
        h.raise_for_status()
    except Exception as exc:
        print(f"ERROR: env not reachable at {args.env_url} — start it with `make dev` first ({exc})",
              file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed_scenarios: list[str] = []
    with args.output.open("w") as f:
        for tid in MISSING_TEMPLATES:
            print(f"\n=== {tid} ===")
            scenarios = [tid] + [f"{tid}__p0{i}" for i in range(1, args.episodes_per_template)]
            for sid in scenarios[: args.episodes_per_template]:
                episode: dict | None = None
                for attempt in range(args.max_attempts_per_scenario):
                    print(f"  collecting {sid} (attempt {attempt + 1}/{args.max_attempts_per_scenario}) ...")
                    episode = collect_episode(
                        env_url=args.env_url,
                        scenario_id=sid,
                        model=args.model,
                        client=client,
                    )
                    if episode and episode["quality_tier"] == "expert":
                        break
                    tier = episode["quality_tier"] if episode else "n/a"
                    print(f"    quality_tier={tier} — {'retrying' if attempt < args.max_attempts_per_scenario - 1 else 'giving up'}")
                if episode:
                    f.write(json.dumps(episode) + "\n")
                    f.flush()
                    written += 1
                    print(f"    saved (final_score={episode['final_score']:.3f}, "
                          f"resolved={episode['incident_resolved']}, "
                          f"steps={episode['steps']})")
                else:
                    failed_scenarios.append(sid)

    print(f"\nWrote {written} episodes to {args.output}")
    if failed_scenarios:
        print(f"Could not collect: {failed_scenarios}")
        print("(Re-run if your model's output stabilises; failures don't block training.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
