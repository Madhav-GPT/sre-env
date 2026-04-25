"""Adapt a chat-completion provider into a runner-compatible policy callable.

The Basic / Advanced / Max runners all expect a ``policy(observation) ->
action_dict`` callable. This module wraps a ``Provider`` in that shape:

1. Render the observation as a plaintext prompt.
2. Call ``provider.chat_sync(...)`` with a system prompt + the observation.
3. Parse the LLM's JSON output.
4. Fall back to ``escalate`` if parsing fails (the env handles unsupported
   actions gracefully).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from sre_gym.exceptions import ActionParseError, ProviderAuthError, ProviderRateLimitError, ProviderModelError
from sre_gym.ui.providers import Provider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_BASIC = """You are a senior SRE on-call agent in the sre-gym environment.

The environment exposes 11 actions:
  query_logs(service)
  query_metrics(service, metric)            metric in [cpu, error_rate, latency]
  query_dependencies(service)
  query_deploys(service)
  rollback_deploy(service)
  restart_service(service)
  isolate_service(service)
  run_check(check_name)                     check_name in [database_recovery, end_to_end]
  submit_hypothesis(hypothesis)             hypothesis is a structured object
  escalate
  declare_resolved

Services live in a 4-node topology: api-gateway / cache / database / worker.

On every turn, output EXACTLY one JSON object — no prose, no markdown fences.

Examples:
  {"action_type":"query_logs","service":"worker"}
  {"action_type":"query_metrics","service":"database","metric":"cpu"}
  {"action_type":"rollback_deploy","service":"worker"}
  {"action_type":"submit_hypothesis","hypothesis":{
    "root_cause":"bad_worker_deploy",
    "affected_services":["worker","database"],
    "confidence":0.8,
    "recommended_next_action":"rollback_deploy"
  }}
  {"action_type":"declare_resolved"}

Diagnose evidence first, remediate carefully, verify with run_check, declare resolved
only after both checks pass. Wrong rollback / premature restart / premature
declare_resolved all carry penalties.
"""

SYSTEM_PROMPT_MAX = SYSTEM_PROMPT_BASIC + """

NOTE: You are running in the Max-tier graph simulator. The same 11 actions
apply against a 22-node service graph (Vercel, Supabase, Stripe, Postgres,
Redis, Kafka + workers and externals). Pick services from the family topology
the observation surfaces.
"""


def _render_basic_observation(obs: Any) -> str:
    """Render a Basic UnifiedIncidentObservation as a prompt string."""
    if hasattr(obs, "prompt_text") and obs.prompt_text:
        return obs.prompt_text
    return str(obs)


def _render_max_observation(obs: Any) -> str:
    """Render the Max graph observation as a prompt string."""
    lines = [
        f"FAMILY: {obs.family_id}",
        f"CHAOS: {obs.chaos}",
        f"TICK: {obs.tick_count}/{obs.max_ticks}",
        f"INCIDENT_SUMMARY: {obs.incident_summary}",
        f"BLAST_RADIUS: {obs.blast_radius}  cause_removed={obs.cause_removed}",
        "",
        "SERVICES:",
    ]
    for sid, svc in obs.services.items():
        if svc["status"] != "healthy":
            lines.append(
                f"- {sid}: {svc['status']} cpu={svc['cpu_pct']:.1f} "
                f"err={svc['error_rate_pct']:.1f} latency={svc['latency_ms']:.1f}"
            )
    if not any(s["status"] != "healthy" for s in obs.services.values()):
        lines.append("- all services healthy")
    lines.append("")
    lines.append(f"LAST_LOG: {obs.last_log}")
    return "\n".join(lines)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from a string."""
    text = (text or "").strip()
    # Strip markdown fences if the model decided to wrap.
    if text.startswith("```"):
        # Drop the fence opener
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    if not text:
        raise ActionParseError(text, "empty response")
    if not text.startswith("{"):
        idx = text.find("{")
        if idx == -1:
            raise ActionParseError(text, "no JSON object found")
        text = text[idx:]
    depth = 0
    end = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ActionParseError(text, "unterminated JSON object")
    try:
        obj = json.loads(text[: end + 1])
    except json.JSONDecodeError as exc:
        raise ActionParseError(text, f"invalid JSON: {exc.msg}") from exc
    if isinstance(obj, dict) and "action_type" not in obj and isinstance(obj.get("action"), str):
        obj = {**obj, "action_type": obj["action"]}
        obj.pop("action", None)
    if not isinstance(obj, dict):
        raise ActionParseError(text, "top-level JSON must be an object")
    return obj


def make_policy(
    provider: Provider,
    *,
    tier: str = "basic",
    on_log: Callable[[str], None] | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Callable[[Any], dict[str, Any]]:
    """Build a ``policy(obs) -> action_dict`` callable backed by a chat provider.

    Parameters
    ----------
    provider
        Any concrete ``Provider`` instance.
    tier
        ``"basic"``, ``"advanced"``, or ``"max"`` — picks the system prompt.
    on_log
        Optional sink for diagnostic messages (provider auth failures, parse fallbacks).
    max_tokens, temperature
        Forwarded to ``provider.chat_sync``.
    """
    system_prompt = SYSTEM_PROMPT_MAX if tier == "max" else SYSTEM_PROMPT_BASIC

    def policy(observation: Any) -> dict[str, Any]:
        # Render the right observation flavour.
        if tier == "max":
            user_text = _render_max_observation(observation)
        else:
            user_text = _render_basic_observation(observation)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            text = provider.chat_sync(messages, max_tokens=max_tokens, temperature=temperature)
        except ProviderAuthError as exc:
            if on_log is not None:
                on_log(f"[provider] {exc} — falling back to escalate()")
            return {"action_type": "escalate"}
        except ProviderRateLimitError as exc:
            if on_log is not None:
                on_log(f"[provider] {exc} — falling back to escalate()")
            return {"action_type": "escalate"}
        except ProviderModelError as exc:
            if on_log is not None:
                on_log(f"[provider] {exc} — falling back to escalate()")
            return {"action_type": "escalate"}

        try:
            return _extract_json_object(text)
        except ActionParseError as exc:
            if on_log is not None:
                on_log(f"[parse] {exc} — falling back to escalate()")
            return {"action_type": "escalate"}

    return policy
