"""One-line local-model CLI for sre-gym.

Drive the Triage tier in-process against a local Ollama model — no API
keys, no HF tokens, no GPU. Useful for hacking on the env on a laptop and
for the README's "5-minute local demo" path.

Usage::

    # 1. Pull a model first (one-time, ~2GB for llama3.2)
    ollama pull llama3.2

    # 2. Run a Triage scenario against it
    python -m sre_gym.local triage worker_deploy_cascade
    python -m sre_gym.local triage gateway_auth_rollout --model qwen2.5:3b
    python -m sre_gym.local triage memory_leak_oom --seed 7 --max-steps 15

The CLI bypasses the FastAPI HTTP layer and drives
``UnifiedIncidentEnvironment`` directly, so a fresh clone with no server
running can produce a complete trace. The Gradio UI and the
``coliseum/`` pool server still go through HTTP — those exist for
multi-tenant and parallel-rollout workloads, which a local CLI doesn't need.

Environment variables:
    OLLAMA_BASE_URL  default ``http://localhost:11434/v1``
    OLLAMA_MODEL     default ``llama3.2`` (overridden by ``--model``)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from sre_gym.exceptions import ProviderModelError
from sre_gym.ui.providers import OllamaProvider


SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no fences.
The 11 actions are:

  query_logs(service)                  query_metrics(service, metric)
  query_dependencies(service)          query_deploys(service)
  rollback_deploy(service)             restart_service(service)
  isolate_service(service)             run_check(check_name)
  submit_hypothesis(hypothesis)        escalate
  declare_resolved

Services live in a 4-node topology: api-gateway / cache / database / worker.
metric in {cpu, error_rate, latency}; check_name in {database_recovery, end_to_end}.

A successful episode looks like: gather evidence -> submit_hypothesis -> rollback ->
restart -> both run_checks pass -> declare_resolved. Wrong rollback / premature
restart / premature declare_resolved are penalized. Idempotent hypotheses score 0.

Respond with one JSON object, e.g. {"action_type": "query_logs", "service": "worker"}.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON-object extraction from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Drop optional language tag on the first line.
        first_newline = text.find("\n")
        if first_newline > 0 and " " not in text[:first_newline]:
            text = text[first_newline + 1 :]
        text = text.strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _action_repr(action_dict: dict[str, Any]) -> str:
    name = action_dict.get("action_type", "?")
    rest = {k: v for k, v in action_dict.items() if k != "action_type"}
    if not rest:
        return name
    if "hypothesis" in rest and isinstance(rest["hypothesis"], dict):
        rc = rest["hypothesis"].get("root_cause", "?")
        rest = {"hypothesis.root_cause": rc}
    body = ", ".join(f"{k}={v}" for k, v in rest.items())
    return f"{name}({body})"


async def run_triage(
    *,
    scenario: str,
    model: str,
    base_url: str,
    seed: int,
    max_steps: int,
) -> int:
    """Drive a Triage episode against an Ollama model. Returns exit code."""
    # Imported lazily so the module imports cleanly even before the env
    # package is built (e.g. during `--help`).
    from unified_incident_env.models import UnifiedIncidentAction
    from unified_incident_env.server.environment import UnifiedIncidentEnvironment

    try:
        provider = OllamaProvider(model=model, base_url=base_url)
    except ProviderModelError as exc:
        print(f"[setup error] {exc}", file=sys.stderr)
        return 2

    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario)
    print(f"[env] reset scenario={scenario} seed={seed} model={model}")
    print(f"[env] {obs.incident_summary}")

    cumulative = 0.0
    for step_idx in range(1, max_steps + 1):
        prompt = obs.prompt_text or ""
        try:
            text = provider.chat_sync(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.0,
            )
        except ProviderModelError as exc:
            print(f"[error] provider error: {exc}", file=sys.stderr)
            return 1

        action_dict = _extract_json(text)
        if action_dict is None:
            print(f"[parse-fail] model output had no JSON object — escalating. raw: {text[:120]!r}")
            action_dict = {"action_type": "escalate"}

        print(f"[action {step_idx:02d}] {_action_repr(action_dict)}")

        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception as exc:
            print(f"[invalid] {exc} — escalating")
            action = UnifiedIncidentAction(action_type="escalate")

        obs = env.step(action)
        cumulative += float(obs.reward)
        sb = obs.score_breakdown or {}
        per_component = (
            f"out={sb.get('outcome', 0.0):.2f} "
            f"valid={sb.get('action_validity', 0.0):.2f} "
            f"fmt={sb.get('format', 0.0):.2f} "
            f"anti={sb.get('anticheat', 0.0):.2f} "
            f"eff={sb.get('efficiency', 0.0):.2f}"
        )
        print(
            f"[reward] Δ={obs.reward:+.3f}  cum={cumulative:+.3f}  "
            f"score={obs.final_score:.3f}  [{per_component}]"
        )

        if obs.done:
            break

    print(
        f"[done] resolved={obs.incident_resolved}  "
        f"steps={obs.tick_count}  final_score={obs.final_score:.3f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sre_gym.local",
        description="Drive an sre-gym tier against a local Ollama model.",
    )
    sub = parser.add_subparsers(dest="tier", required=True)

    triage = sub.add_parser("triage", help="Run a Triage scenario.")
    triage.add_argument("scenario", help="Scenario id, e.g. worker_deploy_cascade")
    triage.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model name (default: llama3.2; or $OLLAMA_MODEL)",
    )
    triage.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        help="Ollama base URL (default: http://localhost:11434/v1)",
    )
    triage.add_argument("--seed", type=int, default=0)
    triage.add_argument("--max-steps", type=int, default=20)

    args = parser.parse_args(argv)

    if args.tier == "triage":
        return asyncio.run(
            run_triage(
                scenario=args.scenario,
                model=args.model,
                base_url=args.base_url,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        )
    parser.error(f"unknown tier: {args.tier}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
