"""Parallel async harness for collecting Claude-driven sre-gym trajectories.

Example:

    python train/collect_trajectories.py \
        --env-url https://dakshdoesdev-sre-gym.hf.space \
        --scenarios worker_deploy_cascade,db_config_rollout,gateway_auth_rollout \
        --models claude-sonnet-4-6,claude-haiku-4-5-20251001 \
        --episodes-per-model 1000 \
        --parallelism 20 \
        --output data/trajectories.jsonl

`--episodes-per-model` is total episodes per model across the resolved scenario
set. Scenario assignment is round-robin so every requested scenario receives
coverage over a long run.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover - handled at runtime in anthropic mode
    AsyncAnthropic = None  # type: ignore[assignment]

from unified_incident_env.client import UnifiedIncidentEnv
from unified_incident_env.models import UnifiedIncidentAction, UnifiedIncidentObservation
from unified_incident_env.server.challenge import SCENARIOS, SUPPORTED_DIFFICULTIES

SYSTEM_PROMPT = (
    "You are collecting trajectories for a deterministic SRE incident benchmark.\n"
    "Return exactly one JSON object and nothing else.\n"
    "Choose only from the allowed action types shown in the prompt.\n"
    "Use only the required fields for the chosen action.\n"
    "Do not include markdown, prose, or code fences."
)
METRIC_OPTIONS = ("cpu", "error_rate", "latency")
CHECK_OPTIONS = ("database_recovery", "end_to_end")
ROOT_CAUSE_OPTIONS = (
    "bad_worker_deploy",
    "database_only_failure",
    "api_gateway_fault",
)


@dataclass(frozen=True)
class EpisodeJob:
    model: str
    scenario_id: str
    ordinal: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-url", required=True, help="sre-gym server base URL")
    parser.add_argument("--scenarios", required=True, help="comma-separated scenario ids, difficulties, or all")
    parser.add_argument("--models", required=True, help="comma-separated Anthropic model ids")
    parser.add_argument("--episodes-per-model", type=int, default=1000)
    parser.add_argument("--parallelism", type=int, default=20)
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument(
        "--driver",
        choices=("anthropic", "fireworks", "groq", "heuristic"),
        default="anthropic",
    )
    parser.add_argument("--anthropic-api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--anthropic-base-url", default=os.getenv("ANTHROPIC_BASE_URL"))
    parser.add_argument("--fireworks-api-key", default=os.getenv("FIREWORKS_API_KEY"))
    parser.add_argument(
        "--fireworks-base-url",
        default=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
    )
    parser.add_argument("--groq-api-key", default=os.getenv("GROQ_API_KEY"))
    parser.add_argument(
        "--groq-base-url",
        default=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--env-timeout-s", type=float, default=45.0)
    parser.add_argument("--anthropic-timeout-s", type=float, default=90.0)
    parser.add_argument("--fireworks-timeout-s", type=float, default=90.0)
    parser.add_argument("--groq-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _split_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolve_scenarios(raw: str) -> list[str]:
    scenario_ids: list[str] = []
    for token in _split_csv(raw):
        if token == "all":
            scenario_ids.extend(SCENARIOS.keys())
            continue
        if token in SUPPORTED_DIFFICULTIES:
            scenario_ids.extend(
                scenario_id
                for scenario_id, scenario in SCENARIOS.items()
                if scenario["difficulty"] == token
            )
            continue
        if token not in SCENARIOS:
            raise SystemExit(f"Unknown scenario selector: {token}")
        scenario_ids.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        if scenario_id not in seen:
            deduped.append(scenario_id)
            seen.add(scenario_id)
    if not deduped:
        raise SystemExit("No scenarios resolved from --scenarios")
    return deduped


def _resolve_models(raw: str) -> list[str]:
    models = _split_csv(raw)
    if not models:
        raise SystemExit("No models resolved from --models")
    return models


def _service_order(observation: UnifiedIncidentObservation) -> list[str]:
    services = list(observation.service_health.items())
    services.sort(
        key=lambda item: (
            item[1].status == "healthy",
            item[1].status == "isolated",
            item[1].error_rate_pct,
            item[1].latency_ms,
        ),
        reverse=True,
    )
    return [name for name, _payload in services]


def _default_action_for_type(action_type: str, observation: UnifiedIncidentObservation) -> dict[str, Any]:
    services = _service_order(observation)
    service = services[0] if services else "database"
    if action_type in {"query_logs", "query_dependencies", "query_deploys", "rollback_deploy", "restart_service", "isolate_service"}:
        return {"action_type": action_type, "service": service}
    if action_type == "query_metrics":
        return {"action_type": action_type, "service": service, "metric": "cpu"}
    if action_type == "run_check":
        pending_checks = [check.name for check in observation.checks if not check.passed]
        check_name = pending_checks[0] if pending_checks else "end_to_end"
        return {"action_type": action_type, "check_name": check_name}
    if action_type == "submit_hypothesis":
        return {
            "action_type": "submit_hypothesis",
            "hypothesis": {
                "root_cause": ROOT_CAUSE_OPTIONS[0],
                "affected_services": services[:2] or ["database"],
                "confidence": 0.5,
                "recommended_next_action": "query_logs",
            },
        }
    return {"action_type": action_type}


def _build_fallback_action(observation: UnifiedIncidentObservation) -> UnifiedIncidentAction:
    pending_checks = [check.name for check in observation.checks if not check.passed]
    if observation.workflow_stage == "validation" and pending_checks:
        return UnifiedIncidentAction(action_type="run_check", check_name=pending_checks[0])
    if observation.workflow_stage == "validation" and not pending_checks:
        return UnifiedIncidentAction(action_type="declare_resolved")
    if observation.workflow_stage == "mitigation":
        services = _service_order(observation)
        service = services[0] if services else "database"
        if "rollback_deploy" in observation.allowed_actions:
            return UnifiedIncidentAction(action_type="rollback_deploy", service=service)
        if "restart_service" in observation.allowed_actions:
            return UnifiedIncidentAction(action_type="restart_service", service=service)
    if "query_logs" in observation.allowed_actions:
        services = _service_order(observation)
        service = services[0] if services else "database"
        return UnifiedIncidentAction(action_type="query_logs", service=service)
    if "query_deploys" in observation.allowed_actions:
        services = _service_order(observation)
        service = services[0] if services else "database"
        return UnifiedIncidentAction(action_type="query_deploys", service=service)
    action_type = observation.allowed_actions[0]
    return UnifiedIncidentAction(**_default_action_for_type(action_type, observation))


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1].strip()
    return text


def _parse_action(raw_text: str, observation: UnifiedIncidentObservation) -> UnifiedIncidentAction | None:
    candidate = _extract_json_object(raw_text)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if "action" in payload and "action_type" not in payload and isinstance(payload["action"], str):
        payload["action_type"] = payload.pop("action")
    if payload.get("action_type") not in observation.allowed_actions:
        return None
    try:
        return UnifiedIncidentAction(**payload)
    except Exception:
        return None


def _build_user_prompt(observation: UnifiedIncidentObservation) -> str:
    required_lines = []
    for action_name, fields in observation.required_fields_by_action.items():
        required_lines.append(
            f"- {action_name}: {', '.join(fields) if fields else '(no extra fields)'}"
        )
    service_names = ", ".join(sorted(observation.service_health))
    return (
        f"{observation.prompt_text}\n\n"
        "JSON_RESPONSE_RULES:\n"
        "- Return exactly one JSON object.\n"
        "- Use only an allowed action_type.\n"
        "- Include only the fields required for that action.\n"
        f"- service must be one of: {service_names}\n"
        f"- metric must be one of: {', '.join(METRIC_OPTIONS)}\n"
        f"- check_name must be one of: {', '.join(CHECK_OPTIONS)}\n"
        f"- hypothesis.root_cause must be one of: {', '.join(ROOT_CAUSE_OPTIONS)}\n"
        "- hypothesis must include root_cause, affected_services, confidence, and recommended_next_action.\n"
        "- Noise alerts are decoys; querying them hurts score.\n\n"
        "REQUIRED_FIELDS_BY_ACTION:\n"
        + "\n".join(required_lines)
    )


def _extract_text_response(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


async def _request_openai_compat_output(
    *,
    http_client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint (Fireworks, Groq).

    Handles 429 rate-limits by honoring the ``Retry-After`` header (or falling
    back to a jittered exponential delay), since the bulk-collection loop
    otherwise saturates the provider's rate limit and cascades to the
    heuristic fallback — which poisons the training dataset.
    """
    attempt = 0
    max_rate_limit_retries = 6
    while True:
        response = await http_client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        if response.status_code == 429 and attempt < max_rate_limit_retries:
            retry_after_raw = response.headers.get("retry-after")
            try:
                delay = float(retry_after_raw) if retry_after_raw else 0.0
            except ValueError:
                delay = 0.0
            if delay <= 0.0:
                delay = min(30.0, (2.0 ** attempt) + random.uniform(0.0, 1.5))
            else:
                delay = min(delay + random.uniform(0.0, 0.75), 60.0)
            await asyncio.sleep(delay)
            attempt += 1
            continue
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()


async def _request_model_output(
    *,
    driver: str,
    anthropic_client: Any,
    openai_compat_http: httpx.AsyncClient | None,
    openai_compat_api_key: str | None,
    openai_compat_base_url: str | None,
    model: str,
    prompt: str,
    fallback_action: UnifiedIncidentAction,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, str | None]:
    if driver == "heuristic":
        return json.dumps(fallback_action.model_dump(exclude_none=True), separators=(",", ":")), "heuristic_driver"
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if driver in ("fireworks", "groq"):
                text = await _request_openai_compat_output(
                    http_client=openai_compat_http,
                    api_key=openai_compat_api_key,
                    base_url=openai_compat_base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            else:
                message = await anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = _extract_text_response(message)
            if text:
                return text, None
            last_error = "empty_text_response"
        except Exception as exc:  # pragma: no cover - exercised in real collection runs
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            await asyncio.sleep(min(2.0 * attempt, 5.0))
    return json.dumps(fallback_action.model_dump(exclude_none=True), separators=(",", ":")), last_error or "model_request_failed"


async def _collect_episode(
    job: EpisodeJob,
    *,
    anthropic_client: Any,
    openai_compat_http: httpx.AsyncClient | None,
    openai_compat_api_key: str | None,
    openai_compat_base_url: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    steps = 0
    async with UnifiedIncidentEnv(base_url=args.env_url) as env:
        observation = (await env.reset(scenario_id=job.scenario_id, episode_id=str(uuid.uuid4()))).observation
        while not observation.done:
            prompt = _build_user_prompt(observation)
            fallback_action = _build_fallback_action(observation)
            response_text, driver_note = await _request_model_output(
                driver=args.driver,
                anthropic_client=anthropic_client,
                openai_compat_http=openai_compat_http,
                openai_compat_api_key=openai_compat_api_key,
                openai_compat_base_url=openai_compat_base_url,
                model=job.model,
                prompt=prompt,
                fallback_action=fallback_action,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
            parsed_action = _parse_action(response_text, observation)
            action = parsed_action or fallback_action
            next_step = await env.step(action)
            next_observation = next_step.observation
            step_failure = next_observation.failure_type
            if parsed_action is None and driver_note is None:
                driver_note = "invalid_model_output"
            if driver_note is not None and action == fallback_action:
                step_failure = step_failure or driver_note
            trajectory.append(
                {
                    "tick": observation.tick_count,
                    "prompt": prompt,
                    "response_text": response_text,
                    "action": action.model_dump(exclude_none=True),
                    "reward": float(next_observation.reward),
                    "tool_output": next_observation.tool_output,
                    "failure_type": step_failure,
                    "workflow_stage": next_observation.workflow_stage,
                }
            )
            observation = next_observation
            steps += 1
    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": job.scenario_id,
        "model": job.model,
        "final_score": float(observation.final_score),
        "incident_resolved": bool(observation.incident_resolved),
        "steps": steps,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "trajectory": trajectory,
    }


async def _worker(
    *,
    name: str,
    jobs: asyncio.Queue[EpisodeJob],
    anthropic_client: Any,
    openai_compat_http: httpx.AsyncClient | None,
    openai_compat_api_key: str | None,
    openai_compat_base_url: str | None,
    args: argparse.Namespace,
    write_lock: asyncio.Lock,
    output_path: Path,
    counters: dict[str, int],
) -> None:
    while True:
        job = await jobs.get()
        try:
            record = await _collect_episode(
                job,
                anthropic_client=anthropic_client,
                openai_compat_http=openai_compat_http,
                openai_compat_api_key=openai_compat_api_key,
                openai_compat_base_url=openai_compat_base_url,
                args=args,
            )
            async with write_lock:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record))
                    handle.write("\n")
            counters["completed"] += 1
            if record["incident_resolved"]:
                counters["resolved"] += 1
            print(
                f"[{counters['completed']}/{counters['total']}] worker={name} model={job.model} "
                f"scenario={job.scenario_id} score={record['final_score']:.3f} "
                f"resolved={str(record['incident_resolved']).lower()} steps={record['steps']}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            jobs.task_done()


async def _run_collection(args: argparse.Namespace) -> None:
    scenario_ids = _resolve_scenarios(args.scenarios)
    models = _resolve_models(args.models)
    if args.driver == "anthropic":
        if AsyncAnthropic is None:
            raise SystemExit("anthropic is not installed. Add it via train/requirements-train.txt before running.")
        if not args.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is required when --driver=anthropic")
    if args.driver == "fireworks":
        if not args.fireworks_api_key:
            raise SystemExit(
                "FIREWORKS_API_KEY is required when --driver=fireworks "
                "(set env var or pass --fireworks-api-key)"
            )
    if args.driver == "groq":
        if not args.groq_api_key:
            raise SystemExit(
                "GROQ_API_KEY is required when --driver=groq "
                "(set env var or pass --groq-api-key)"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    jobs: asyncio.Queue[EpisodeJob] = asyncio.Queue()
    for model in models:
        for ordinal in range(args.episodes_per_model):
            scenario_id = scenario_ids[ordinal % len(scenario_ids)]
            jobs.put_nowait(EpisodeJob(model=model, scenario_id=scenario_id, ordinal=ordinal))

    probe_client = httpx.AsyncClient(
        base_url=args.env_url.rstrip("/"),
        timeout=httpx.Timeout(args.env_timeout_s),
        follow_redirects=True,
    )
    health = await probe_client.get("/health")
    health.raise_for_status()
    await probe_client.aclose()

    anthropic_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(args.anthropic_timeout_s),
        limits=httpx.Limits(
            max_connections=max(args.parallelism * 2, 20),
            max_keepalive_connections=max(args.parallelism, 10),
        ),
        follow_redirects=True,
    )
    anthropic_client = None
    if args.driver == "anthropic":
        anthropic_client = AsyncAnthropic(
            api_key=args.anthropic_api_key,
            base_url=args.anthropic_base_url or None,
            http_client=anthropic_http_client,
        )

    openai_compat_http: httpx.AsyncClient | None = None
    openai_compat_api_key: str | None = None
    openai_compat_base_url: str | None = None
    if args.driver in ("fireworks", "groq"):
        if args.driver == "fireworks":
            timeout_s = args.fireworks_timeout_s
            openai_compat_api_key = args.fireworks_api_key
            openai_compat_base_url = args.fireworks_base_url
        else:
            timeout_s = args.groq_timeout_s
            openai_compat_api_key = args.groq_api_key
            openai_compat_base_url = args.groq_base_url
        openai_compat_http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(
                max_connections=max(args.parallelism * 2, 20),
                max_keepalive_connections=max(args.parallelism, 10),
            ),
            follow_redirects=True,
        )

    write_lock = asyncio.Lock()
    counters = {
        "completed": 0,
        "resolved": 0,
        "total": jobs.qsize(),
    }
    workers = [
        asyncio.create_task(
            _worker(
                name=f"w{index + 1}",
                jobs=jobs,
                anthropic_client=anthropic_client,
                openai_compat_http=openai_compat_http,
                openai_compat_api_key=openai_compat_api_key,
                openai_compat_base_url=openai_compat_base_url,
                args=args,
                write_lock=write_lock,
                output_path=output_path,
                counters=counters,
            )
        )
        for index in range(min(args.parallelism, counters["total"]))
    ]

    try:
        await jobs.join()
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await anthropic_http_client.aclose()
        if openai_compat_http is not None:
            await openai_compat_http.aclose()

    success_rate = counters["resolved"] / counters["total"] if counters["total"] else 0.0
    print(
        f"completed={counters['completed']} resolved={counters['resolved']} "
        f"success_rate={success_rate:.3f} output={output_path}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(_run_collection(args))


if __name__ == "__main__":
    main()
