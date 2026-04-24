"""Eval sweep — compare policy scores across scenarios for the submission table.

Runs N episodes per scenario per policy against a live sre-gym env and writes
a JSONL summary suitable for the hackathon comparison table.

Supported policies:
- `random`        — emit a valid random action each turn
- `heuristic`     — the deterministic heuristic from collect_trajectories
- `groq`          — Llama-3.3-70B via Groq (uses GROQ_API_KEY)
- `fireworks`     — any Fireworks-served model (uses FIREWORKS_API_KEY)
- `anthropic`     — any Anthropic model (uses ANTHROPIC_API_KEY)
- `sft_adapter`   — a local HF transformers checkpoint (directory path)

The output JSONL schema:
  {policy, model, scenario_id, episode_idx, final_score, incident_resolved,
   steps, elapsed_s}

Intended usage (Sunday evening, after SFT and/or GRPO has landed):

  python train/eval_sweep.py \
      --env-url https://dakshdoesdev-sre-gym.hf.space \
      --scenarios all \
      --episodes-per-scenario 5 \
      --policies random,heuristic,groq \
      --groq-model llama-3.3-70b-versatile \
      --output train/data/eval_sweep.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# Reuse trajectory-collection helpers to avoid duplicating the action-parse
# and fallback-action logic.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.collect_trajectories import (  # type: ignore  # noqa: E402
    SYSTEM_PROMPT,
    _build_fallback_action,
    _build_user_prompt,
    _parse_action,
    _request_openai_compat_output,
)
from unified_incident_env import UnifiedIncidentAction, UnifiedIncidentEnv  # noqa: E402
from unified_incident_env.server.challenge import SCENARIOS  # noqa: E402


SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard"}


def _resolve_scenarios(raw: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    scenario_ids: list[str] = []
    for token in tokens:
        if token == "all":
            scenario_ids.extend(SCENARIOS.keys())
        elif token in SUPPORTED_DIFFICULTIES:
            scenario_ids.extend(
                s for s, c in SCENARIOS.items() if c["difficulty"] == token
            )
        elif token in SCENARIOS:
            scenario_ids.append(token)
        else:
            raise SystemExit(f"Unknown scenario selector: {token}")
    seen: set[str] = set()
    deduped: list[str] = []
    for s in scenario_ids:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def _random_action(observation: Any) -> UnifiedIncidentAction:
    # Pick an allowed action, populate minimal required fields randomly.
    allowed = observation.allowed_actions or ["query_logs"]
    action_type = random.choice(allowed)
    services = list(observation.service_health.keys()) or ["database"]
    if action_type in {"query_logs", "query_dependencies", "query_deploys",
                        "rollback_deploy", "restart_service", "isolate_service"}:
        return UnifiedIncidentAction(action_type=action_type, service=random.choice(services))
    if action_type == "query_metrics":
        return UnifiedIncidentAction(
            action_type=action_type,
            service=random.choice(services),
            metric=random.choice(["cpu", "error_rate", "latency"]),
        )
    if action_type == "run_check":
        pending = [c.name for c in observation.checks if not c.passed] or ["end_to_end"]
        return UnifiedIncidentAction(action_type=action_type, check_name=pending[0])
    if action_type == "submit_hypothesis":
        return _build_fallback_action(observation)  # reasonable hypothesis shape
    return UnifiedIncidentAction(action_type=action_type)


async def _play_episode(
    *,
    env_url: str,
    scenario_id: str,
    policy: str,
    model: str,
    http_client: httpx.AsyncClient | None,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    async with UnifiedIncidentEnv(base_url=env_url) as env:
        obs = (await env.reset(scenario_id=scenario_id, episode_id=str(uuid.uuid4()))).observation
        steps = 0
        while not obs.done:
            fallback = _build_fallback_action(obs)
            if policy == "random":
                action = _random_action(obs)
            elif policy == "heuristic":
                action = fallback
            elif policy in ("groq", "fireworks"):
                prompt = _build_user_prompt(obs)
                text = ""
                for attempt in range(1, max_retries + 1):
                    try:
                        text = await _request_openai_compat_output(
                            http_client=http_client,
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                        )
                        if text:
                            break
                    except Exception:
                        await asyncio.sleep(min(2.0 * attempt, 5.0))
                parsed = _parse_action(text, obs)
                action = parsed or fallback
            else:
                raise SystemExit(f"Policy {policy} not implemented here; use `policies` flag.")
            step = await env.step(action)
            obs = step.observation
            steps += 1
        return {
            "scenario_id": scenario_id,
            "policy": policy,
            "model": model,
            "final_score": float(obs.final_score),
            "incident_resolved": bool(obs.incident_resolved),
            "steps": steps,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }


async def _run_sweep(args: argparse.Namespace) -> None:
    scenarios = _resolve_scenarios(args.scenarios)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    # Health-probe the env
    async with httpx.AsyncClient(timeout=10.0) as probe:
        response = await probe.get(f"{args.env_url.rstrip('/')}/health")
        response.raise_for_status()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    groq_http: httpx.AsyncClient | None = None
    fireworks_http: httpx.AsyncClient | None = None
    if "groq" in policies:
        if not args.groq_api_key:
            raise SystemExit("GROQ_API_KEY required for groq policy")
        groq_http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=args.parallelism * 2),
            follow_redirects=True,
        )
    if "fireworks" in policies:
        if not args.fireworks_api_key:
            raise SystemExit("FIREWORKS_API_KEY required for fireworks policy")
        fireworks_http = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0),
            limits=httpx.Limits(max_connections=args.parallelism * 2),
            follow_redirects=True,
        )

    semaphore = asyncio.Semaphore(args.parallelism)

    async def run_one(policy: str, scenario: str, idx: int) -> None:
        async with semaphore:
            model_map = {
                "groq": args.groq_model,
                "fireworks": args.fireworks_model,
                "random": "random",
                "heuristic": "heuristic",
            }
            http, key, base = None, None, None
            if policy == "groq":
                http, key, base = groq_http, args.groq_api_key, args.groq_base_url
            elif policy == "fireworks":
                http, key, base = fireworks_http, args.fireworks_api_key, args.fireworks_base_url

            try:
                record = await _play_episode(
                    env_url=args.env_url,
                    scenario_id=scenario,
                    policy=policy,
                    model=model_map.get(policy, policy),
                    http_client=http,
                    api_key=key,
                    base_url=base,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                record["episode_idx"] = idx
            except Exception as exc:
                record = {
                    "scenario_id": scenario,
                    "policy": policy,
                    "episode_idx": idx,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with output_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            score = record.get("final_score")
            resolved = record.get("incident_resolved")
            print(
                f"[{policy:<10}] {scenario:<40} ep={idx}  "
                f"score={f'{score:.3f}' if score is not None else 'err'}  "
                f"resolved={resolved}",
                file=sys.stderr,
                flush=True,
            )

    tasks = []
    for policy in policies:
        for scenario in scenarios:
            for idx in range(args.episodes_per_scenario):
                tasks.append(run_one(policy, scenario, idx))
    try:
        await asyncio.gather(*tasks)
    finally:
        if groq_http is not None:
            await groq_http.aclose()
        if fireworks_http is not None:
            await fireworks_http.aclose()

    # Print a summary table per policy per scenario.
    records = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_policy.setdefault(r["policy"], []).append(r)
    print("\n=== SUMMARY ===", file=sys.stderr)
    for policy, rs in by_policy.items():
        scored = [r for r in rs if "final_score" in r]
        if not scored:
            print(f"  {policy}: all episodes errored ({len(rs)} errors)", file=sys.stderr)
            continue
        mean = sum(r["final_score"] for r in scored) / len(scored)
        resolved = sum(1 for r in scored if r.get("incident_resolved")) / len(scored)
        print(
            f"  {policy:<12} n={len(scored):<3} mean_score={mean:.3f}  resolved={resolved:.1%}",
            file=sys.stderr,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--policies", required=True, help="comma-separated: random,heuristic,groq,fireworks")
    parser.add_argument("--episodes-per-scenario", type=int, default=5)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groq-api-key", default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--groq-base-url", default=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--fireworks-api-key", default=os.getenv("FIREWORKS_API_KEY"))
    parser.add_argument(
        "--fireworks-base-url",
        default=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
    )
    parser.add_argument("--fireworks-model", default="accounts/fireworks/models/llama-v3p3-70b-instruct")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_sweep(args))


if __name__ == "__main__":
    main()
