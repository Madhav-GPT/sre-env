"""Streaming runner — async generator producing terminal-style trace lines.

Per UI Build Addendum §3.5–§3.7. Each invocation yields *cumulative* terminal
text on every step so Gradio's streaming-generator pattern can render the
trace incrementally without re-rendering the whole pane.

Three tier-specific paths:

- **Basic**     — calls ``/reset`` and ``/step`` against the in-process FastAPI
                  app via httpx, so the demo trace looks like a real client
                  driving the OpenEnv contract. Per §3.6 we explicitly do not
                  bypass the HTTP layer.
- **Advanced**  — invokes ``sre_gym.advanced.runner.run_advanced``.
- **Max**       — invokes ``sre_gym.max.runner.run_max``.

Each line is ≤ 120 chars (long observations are truncated with " …").

Per §3.7, provider auth / parse failures surface as red lines inside the
trace; we never raise to a Gradio toast, so every artifact stays in one
readable terminal block.

CLI entry-point::

    python -m sre_gym.ui.runner --tier basic --template worker_deploy_cascade \
        --seed 42 --provider dummy
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Any, AsyncIterator, Callable, Iterable

from sre_gym.exceptions import (
    ActionParseError,
    ProviderAuthError,
    ProviderModelError,
    ProviderRateLimitError,
    SREGymError,
)
from sre_gym.ui.policies import _extract_json_object
from sre_gym.ui.providers import DummyProvider, Provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal formatting helpers.
# ---------------------------------------------------------------------------


_LINE_WIDTH = 120


def _ts(start: float) -> str:
    """Return MM:SS elapsed since ``start`` (epoch seconds)."""
    delta = max(0.0, time.time() - start)
    minutes = int(delta // 60)
    seconds = int(delta % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _truncate(text: str, width: int = _LINE_WIDTH) -> str:
    text = (text or "").replace("\n", " ⏎ ").replace("\t", " ")
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _line(start: float, prefix: str, body: str) -> str:
    return f"[{_ts(start)}] {prefix:<7}: {_truncate(body, _LINE_WIDTH - 12)}"


# ---------------------------------------------------------------------------
# Per-component breakdown projection (UI acceptance criterion).
# ---------------------------------------------------------------------------


def _project_breakdown(score_breakdown: dict[str, float]) -> dict[str, float]:
    """Project the env's 7-dim rubric into the 5 surface dimensions the UI shows.

    Returns a dict with keys ``outcome / valid / fmt / anti / eff``:

    - ``outcome``  = recovery_score + impact_score
    - ``valid``    = containment_score + verification_score
    - ``fmt``      = 1.0 if no parse failures occurred during the episode
                     (the UI uses ``runner_format_score`` to flip this to <1.0
                     on parse-fail; default is 1.0 since the env validates JSON
                     at the boundary)
    - ``anti``     = noise_handling_score
    - ``eff``      = efficiency_score + speed_bonus
    """
    sb = score_breakdown or {}
    return {
        "outcome": round(sb.get("recovery_score", 0.0) + sb.get("impact_score", 0.0), 3),
        "valid": round(sb.get("containment_score", 0.0) + sb.get("verification_score", 0.0), 3),
        "fmt": float(sb.get("runner_format_score", 1.0)),
        "anti": round(sb.get("noise_handling_score", 0.0), 3),
        "eff": round(sb.get("efficiency_score", 0.0) + sb.get("speed_bonus", 0.0), 3),
    }


def _format_breakdown(b: dict[str, float]) -> str:
    return (
        f"outcome={b['outcome']:.2f} valid={b['valid']:.2f} "
        f"fmt={b['fmt']:.2f} anti={b['anti']:.2f} eff={b['eff']:.2f}"
    )


# ---------------------------------------------------------------------------
# Provider → action helper.
# ---------------------------------------------------------------------------


async def _ask_provider(
    provider: Provider,
    system_prompt: str,
    observation_text: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send (system, user) messages to the provider, parse the action JSON.

    Returns ``(action_dict, error_message)``. ``error_message`` is non-None
    only when the call failed; the caller renders it in red.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation_text},
    ]
    try:
        text = await provider.chat(messages, max_tokens=max_tokens, temperature=temperature)
    except ProviderAuthError as exc:
        return None, str(exc)
    except ProviderRateLimitError as exc:
        return None, str(exc)
    except ProviderModelError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"provider error: {exc}"

    try:
        return _extract_json_object(text), None
    except ActionParseError as exc:
        return None, f"parse failure: {exc}"


# ---------------------------------------------------------------------------
# Basic-tier streamer (calls the live FastAPI /reset + /step).
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_BASIC = """You are a senior SRE on-call agent inside the sre-gym environment.

Output EXACTLY one JSON object per turn — no prose, no markdown. The 11 actions are:

  query_logs(service)                  query_metrics(service, metric)
  query_dependencies(service)          query_deploys(service)
  rollback_deploy(service)             restart_service(service)
  isolate_service(service)             run_check(check_name)
  submit_hypothesis(hypothesis)        escalate
  declare_resolved

Services live in a 4-node topology: api-gateway / cache / database / worker.
metric ∈ {cpu, error_rate, latency}; check_name ∈ {database_recovery, end_to_end}.

A successful episode looks like: gather evidence → submit_hypothesis → rollback →
restart → both run_checks pass → declare_resolved. Wrong rollback / premature
restart / premature declare_resolved are penalized. Idempotent hypotheses score 0.
"""


async def stream_basic_episode(
    *,
    template: str,
    seed: int,
    provider: Provider,
    base_url: str = "http://127.0.0.1:7860",
    max_steps: int = 20,
    on_log: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Run a Basic episode through HTTP /reset + /step calls.

    Yields cumulative terminal text. Per §3.6 we deliberately drive the env
    over HTTP so the demo trace looks identical to what an external TRL
    client would produce.
    """
    import httpx

    start = time.time()
    transcript: list[str] = []

    def emit(prefix: str, body: str) -> str:
        line = _line(start, prefix, body)
        transcript.append(line)
        if on_log is not None:
            on_log(line)
        return "\n".join(transcript)

    # Greeting + reset
    yield emit("env", f"reset scenario={template} seed={seed} (POST /reset)")
    async with httpx.AsyncClient(base_url=base_url, timeout=20.0) as client:
        try:
            reset_resp = await client.post("/reset", json={"scenario_id": template})
        except httpx.HTTPError as exc:
            yield emit("ERROR", f"reset failed: {exc}")
            return
        if reset_resp.status_code != 200:
            yield emit("ERROR", f"reset returned HTTP {reset_resp.status_code}")
            return
        reset_body = reset_resp.json()
        observation = reset_body.get("observation", reset_body)

        cumulative_reward = 0.0
        format_failures = 0
        ticks_taken = 0
        score_breakdown: dict[str, float] = {}
        incident_resolved = False
        done = False

        for step_idx in range(1, max_steps + 1):
            obs_text = observation.get("prompt_text") or "(no prompt_text)"
            action_dict, err = await _ask_provider(provider, SYSTEM_PROMPT_BASIC, obs_text)
            if err is not None:
                # Surface in red — the Gradio CSS class .terminal renders ERROR lines red.
                yield emit("ERROR", err)
                # Still escalate so the run terminates cleanly rather than hanging.
                action_dict = {"action_type": "escalate"}
                format_failures += 1

            yield emit("action", _action_repr(action_dict))

            try:
                step_resp = await client.post("/step", json={"action": action_dict})
            except httpx.HTTPError as exc:
                yield emit("ERROR", f"step failed: {exc}")
                break
            if step_resp.status_code != 200:
                yield emit("ERROR", f"step returned HTTP {step_resp.status_code}: {step_resp.text[:80]}")
                break
            body = step_resp.json()
            observation = body.get("observation", body)
            reward_delta = float(body.get("reward", 0.0))
            cumulative_reward += reward_delta
            ticks_taken = int(observation.get("tick_count", step_idx))
            score_breakdown = dict(observation.get("score_breakdown") or {})
            incident_resolved = bool(observation.get("incident_resolved"))
            done = bool(body.get("done"))

            tool_output = observation.get("tool_output") or observation.get("last_action_result") or "(no output)"
            yield emit("obs", tool_output)
            yield emit(
                "reward",
                f"Δ={reward_delta:+.3f}  cum={cumulative_reward:+.3f}  score={observation.get('final_score', 0.0):.3f}",
            )
            if done:
                break

    # Final summary
    if format_failures > 0:
        score_breakdown["runner_format_score"] = max(0.0, 1.0 - 0.2 * format_failures)
    breakdown = _project_breakdown(score_breakdown)
    final_score = float(observation.get("final_score", 0.0)) if isinstance(observation, dict) else 0.0
    summary_line = (
        f"DONE  reward={final_score:.3f}  resolved={incident_resolved}  "
        f"steps={ticks_taken}  [{_format_breakdown(breakdown)}]"
    )
    yield emit("done", summary_line)


def _action_repr(action_dict: dict[str, Any]) -> str:
    """Render an action dict as ``action_type(arg=value, ...)``."""
    name = action_dict.get("action_type", "?")
    rest = {k: v for k, v in action_dict.items() if k != "action_type"}
    if not rest:
        return name
    if "hypothesis" in rest and isinstance(rest["hypothesis"], dict):
        rc = rest["hypothesis"].get("root_cause", "?")
        rest = {"hypothesis.root_cause": rc}
    body = ", ".join(f"{k}={v}" for k, v in rest.items())
    return f"{name}({body})"


# ---------------------------------------------------------------------------
# Advanced + Max streamers (delegate to existing runners).
# ---------------------------------------------------------------------------


async def stream_advanced_episode(
    *,
    scenario: str,
    seed: int,
    provider: Provider | None,
    on_log: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Run an Advanced scenario via ``sre_gym.advanced.runner.run_advanced``.

    The Advanced runner is sync; we drive it on a worker thread and tee its
    log lines into the streaming trace.
    """
    from sre_gym.advanced.runner import run_advanced
    from sre_gym.ui.policies import make_policy

    start = time.time()
    transcript: list[str] = []
    queue: asyncio.Queue[str] = asyncio.Queue()
    sentinel = "__END__"

    def push(prefix: str, body: str) -> None:
        line = _line(start, prefix, body)
        transcript.append(line)
        try:
            queue.put_nowait(line)
        except asyncio.QueueFull:  # pragma: no cover
            pass
        if on_log is not None:
            on_log(line)

    def runner_log(line: str) -> None:
        push("phase", line)

    policy = None
    if provider is not None:
        policy = make_policy(provider, tier="basic")

    def runner_thread() -> None:
        try:
            result = run_advanced(scenario, policy=policy, seed=seed, on_log=runner_log)
            push(
                "done",
                f"final={result.final_reward:.3f}  decay×{result.horizon_decay_factor:.3f}  "
                f"phases={len(result.phases)}  success={result.success}",
            )
        except SREGymError as exc:
            push("ERROR", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            push("ERROR", f"advanced runner crashed: {exc}")
        finally:
            queue.put_nowait(sentinel)

    push("env", f"start advanced scenario={scenario} seed={seed}")
    yield "\n".join(transcript)

    loop = asyncio.get_event_loop()
    fut = loop.run_in_executor(None, runner_thread)
    try:
        while True:
            line = await queue.get()
            if line == sentinel:
                break
            yield "\n".join(transcript)
    finally:
        await fut


async def stream_max_episode(
    *,
    family: str,
    chaos: str,
    seed: int,
    provider: Provider | None,
    on_log: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Run a Max episode via ``sre_gym.max.runner.run_max``."""
    from sre_gym.max.runner import run_max
    from sre_gym.ui.policies import make_policy

    start = time.time()
    transcript: list[str] = []
    queue: asyncio.Queue[str] = asyncio.Queue()
    sentinel = "__END__"

    def push(prefix: str, body: str) -> None:
        line = _line(start, prefix, body)
        transcript.append(line)
        try:
            queue.put_nowait(line)
        except asyncio.QueueFull:  # pragma: no cover
            pass
        if on_log is not None:
            on_log(line)

    def runner_log(line: str) -> None:
        push("chaos", line)

    policy = None
    if provider is not None:
        policy = make_policy(provider, tier="max")

    def runner_thread() -> None:
        try:
            result = run_max(family, chaos=chaos, policy=policy, seed=seed, on_log=runner_log)
            sec = " [SECURITY]" if result.classification == "security" else ""
            push(
                "done",
                f"final={result.final_reward:.3f}{sec}  resolved={result.incident_resolved}  "
                f"blast={result.blast_radius}",
            )
        except SREGymError as exc:
            push("ERROR", str(exc))
        except Exception as exc:  # pragma: no cover
            push("ERROR", f"max runner crashed: {exc}")
        finally:
            queue.put_nowait(sentinel)

    push("env", f"start max family={family} chaos={chaos} seed={seed}")
    yield "\n".join(transcript)

    loop = asyncio.get_event_loop()
    fut = loop.run_in_executor(None, runner_thread)
    try:
        while True:
            line = await queue.get()
            if line == sentinel:
                break
            yield "\n".join(transcript)
    finally:
        await fut


# ---------------------------------------------------------------------------
# Tier dispatcher.
# ---------------------------------------------------------------------------


async def stream_episode(
    *,
    tier: str,
    template: str | None = None,
    scenario: str | None = None,
    family: str | None = None,
    chaos: str = "deploy_regression",
    seed: int = 0,
    provider: Provider | None = None,
    base_url: str = "http://127.0.0.1:7860",
    max_steps: int = 20,
    on_log: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Tier-aware streaming entry-point used by the Gradio UI."""
    if tier == "basic":
        if provider is None:
            provider = DummyProvider()
        async for chunk in stream_basic_episode(
            template=template or "worker_deploy_cascade",
            seed=seed,
            provider=provider,
            base_url=base_url,
            max_steps=max_steps,
            on_log=on_log,
        ):
            yield chunk
    elif tier == "advanced":
        async for chunk in stream_advanced_episode(
            scenario=scenario or "cascading_release_train",
            seed=seed,
            provider=provider,
            on_log=on_log,
        ):
            yield chunk
    elif tier == "max":
        async for chunk in stream_max_episode(
            family=family or "ecommerce_vibecoded_saas",
            chaos=chaos,
            seed=seed,
            provider=provider,
            on_log=on_log,
        ):
            yield chunk
    else:
        raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# CLI entry-point — used by Stage U4 done-criterion smoke check.
# ---------------------------------------------------------------------------


def _make_cli_provider(name: str) -> Provider:
    if name == "dummy":
        return DummyProvider()
    raise ValueError(f"CLI smoke supports --provider dummy only (got {name!r})")


async def _cli_main(args: argparse.Namespace) -> int:
    provider = _make_cli_provider(args.provider) if args.provider else None
    async for trace in stream_episode(
        tier=args.tier,
        template=args.template,
        scenario=args.scenario,
        family=args.family,
        chaos=args.chaos,
        seed=args.seed,
        provider=provider,
        base_url=args.base_url,
        max_steps=args.max_steps,
        on_log=lambda line: print(line, flush=True),
    ):
        # Trace is cumulative; we only want the per-line log we emitted via on_log.
        # The async generator already pushes lines to stdout via the on_log callback.
        pass
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sre_gym.ui.runner", description=__doc__)
    parser.add_argument("--tier", choices=["basic", "advanced", "max"], default="basic")
    parser.add_argument("--template", default="worker_deploy_cascade")
    parser.add_argument("--scenario", default="cascading_release_train")
    parser.add_argument("--family", default="ecommerce_vibecoded_saas")
    parser.add_argument("--chaos", default="deploy_regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--provider", default="dummy", help="dummy | (any other name will be rejected)")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return asyncio.run(_cli_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
