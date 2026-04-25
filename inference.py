#!/usr/bin/env python3
"""Submission inference script for the honest narrow incident environment."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from unified_incident_env.client import UnifiedIncidentEnv
from unified_incident_env.models import UnifiedIncidentAction, UnifiedIncidentObservation
from unified_incident_env.server.challenge import DEFAULT_SCENARIO_ID, SCENARIOS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or UnifiedIncidentEnv.DEFAULT_BASE_URL
ENV_NAME = "unified-incident-env"
MAX_TOKENS = 260


def create_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}", flush=True)


def _service_order(observation: UnifiedIncidentObservation) -> list[str]:
    services = list(observation.service_health.items())
    services.sort(key=lambda item: (item[1].status != "crashed", item[1].status != "degraded", -item[1].error_rate_pct))
    return [name for name, _payload in services]


def _heuristic_root_cause(observation: UnifiedIncidentObservation) -> tuple[str, list[str]]:
    """Pick a plausible root cause + affected services from the observation.

    Decision tree maps the loudest signal to the most likely root cause across
    all 12 templates. This is the Stage-1 fallback — when the LLM is offline
    or returns malformed JSON, the heuristic still produces a calibrated
    hypothesis good enough to score partial credit on the 5-component rubric.

    Each branch corresponds to one of the 12 RootCauseType enum values.
    """
    summary = (observation.incident_summary or "").lower()
    services = observation.service_health
    db = services.get("database")
    cache = services.get("cache")
    worker = services.get("worker")
    gateway = services.get("api-gateway")

    # Round-2 templates first (the new failure modes)
    if "memory" in summary or "oom" in summary or "restart loop" in summary:
        return "memory_leak_runaway", ["worker", "database", "api-gateway"]
    if "token" in summary or "credential" in summary or "auth" in summary and "401" in summary:
        return "credential_rotation_breakage", ["worker", "api-gateway"]
    if "dns" in summary or "discovery" in summary or "partition" in summary:
        return "network_dns_partition", ["cache", "worker", "api-gateway"]
    if "retry" in summary or "rate limit" in summary or "429" in summary:
        return "external_rate_limit_storm", ["worker", "database", "api-gateway"]
    if "lock" in summary or "migration" in summary and "concurrently" in summary:
        return "migration_lock_contention", ["database", "worker", "api-gateway"]
    if "maxclients" in summary or "pool" in summary and "cache" in summary:
        return "dependency_pool_exhausted", ["cache", "worker", "api-gateway"]

    # Vibe-coded SaaS extension band
    if "stripe" in summary or "webhook" in summary:
        return "payment_webhook_regression", ["api-gateway", "database"]
    if "schema" in summary or "prisma" in summary or "plan_tier" in summary:
        return "schema_migration_mismatch", ["api-gateway", "worker", "database"]
    if "ttl" in summary or "stale" in summary or "session" in summary and "cross" in summary:
        return "cache_ttl_regression", ["cache", "api-gateway"]

    # v2 catalogue (default fallbacks based on which service is loudest)
    if gateway and gateway.error_rate_pct >= 30 and (db is None or db.error_rate_pct < 5):
        return "api_gateway_fault", ["api-gateway", "worker"]
    if db and db.status == "degraded" and (worker is None or worker.status != "crashed"):
        return "database_only_failure", ["database", "api-gateway", "worker"]
    return "bad_worker_deploy", ["worker", "database", "api-gateway"]


def _default_action_for_type(action_type: str, observation: UnifiedIncidentObservation) -> dict[str, Any]:
    services = _service_order(observation)
    service = services[0] if services else "database"
    if action_type in {"query_logs", "query_dependencies", "query_deploys", "rollback_deploy", "restart_service", "isolate_service"}:
        if action_type == "rollback_deploy":
            service = "worker"
        return {"action_type": action_type, "service": service}
    if action_type == "query_metrics":
        return {"action_type": action_type, "service": service, "metric": "cpu"}
    if action_type == "run_check":
        check_name = "database_recovery"
        if observation.service_health.get("database") and observation.service_health["database"].status == "healthy":
            check_name = "end_to_end"
        return {"action_type": action_type, "check_name": check_name}
    if action_type == "submit_hypothesis":
        root_cause, affected = _heuristic_root_cause(observation)
        return {
            "action_type": "submit_hypothesis",
            "hypothesis": {
                "root_cause": root_cause,
                "affected_services": affected,
                "confidence": 0.6,
                "recommended_next_action": "rollback_deploy",
            },
        }
    return {"action_type": action_type}


def parse_action(raw: str, observation: UnifiedIncidentObservation) -> UnifiedIncidentAction | None:
    text = raw.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "action" in data and "action_type" not in data and isinstance(data["action"], str):
        data = {**data, "action_type": data["action"]}
        data.pop("action", None)
    action_type = data.get("action_type")
    if action_type not in observation.allowed_actions:
        return None
    try:
        return UnifiedIncidentAction(**data)
    except Exception:
        return None


def build_user_prompt(observation: UnifiedIncidentObservation) -> str:
    required_lines = []
    for action, fields in observation.required_fields_by_action.items():
        required_lines.append(f"- {action}: {', '.join(fields) if fields else '(no extra fields)'}")
    checks = "\n".join(
        f"- {check.name}: {'passed' if check.passed else 'pending'} - {check.detail}"
        for check in observation.checks
    ) or "- none"
    return (
        "Return exactly one JSON object representing the next action.\n"
        f"Current stage: {observation.workflow_stage}\n"
        f"Incident summary: {observation.incident_summary}\n"
        f"Current score: {observation.final_score:.4f}\n"
        f"Last action result: {observation.last_action_result or 'none'}\n"
        f"Tool output: {observation.tool_output or 'none'}\n"
        f"Failure: {observation.failure_type or 'none'}\n"
        f"Why failed: {observation.why_failed or 'none'}\n"
        f"User impact: {observation.user_impact:.2f}\n"
        f"SLO burn rate: {observation.slo_burn_rate:.2f}\n"
        "Allowed actions:\n"
        + "\n".join(f"- {action}" for action in observation.allowed_actions)
        + "\nRequired fields:\n"
        + "\n".join(required_lines)
        + "\nChecks:\n"
        + checks
    )


_ROOT_CAUSE_ENUM: list[str] = [
    # Original 3 (v2 catalogue)
    "bad_worker_deploy",
    "database_only_failure",
    "api_gateway_fault",
    # Vibe-coded SaaS extension band
    "payment_webhook_regression",
    "schema_migration_mismatch",
    "cache_ttl_regression",
    # Round-2 Basic-tier additions (April 2026 hackathon)
    "dependency_pool_exhausted",
    "memory_leak_runaway",
    "credential_rotation_breakage",
    "network_dns_partition",
    "external_rate_limit_storm",
    "migration_lock_contention",
]
"""All 12 root causes the model can hypothesize.

Mirrors ``unified_incident_env.models.RootCauseType`` exactly. Kept as a module-
level constant so a CI test can import and diff this against the Literal
without round-tripping through reflection.
"""


def _schema(observation: UnifiedIncidentObservation) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action_type": {"type": "string", "enum": observation.allowed_actions},
        "service": {"type": "string", "enum": sorted(observation.service_health)},
        "metric": {"type": "string", "enum": ["cpu", "error_rate", "latency"]},
        "check_name": {"type": "string", "enum": ["database_recovery", "end_to_end"]},
        "hypothesis": {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string", "enum": list(_ROOT_CAUSE_ENUM)},
                "affected_services": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(observation.service_health)},
                    "minItems": 1,
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "recommended_next_action": {
                    "type": "string",
                    "enum": [
                        "query_logs",
                        "query_metrics",
                        "query_dependencies",
                        "query_deploys",
                        "rollback_deploy",
                        "restart_service",
                        "run_check",
                        "isolate_service",
                        "escalate",
                        "declare_resolved",
                    ],
                },
            },
            "required": ["root_cause", "affected_services", "confidence", "recommended_next_action"],
            "additionalProperties": False,
        },
    }
    required = ["action_type"]
    for action, fields in observation.required_fields_by_action.items():
        if action in observation.allowed_actions:
            for field in fields:
                if field not in required:
                    required.append(field)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def request_action(client: OpenAI, observation: UnifiedIncidentObservation) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an incident responder. Respond with JSON only."},
            {"role": "user", "content": build_user_prompt(observation)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "incident_action",
                "strict": True,
                "schema": _schema(observation),
            },
        },
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return (completion.choices[0].message.content or "").strip()


def build_fallback_action(observation: UnifiedIncidentObservation) -> UnifiedIncidentAction:
    services = _service_order(observation)
    if "query_deploys" in observation.allowed_actions and "worker" in observation.service_health:
        return UnifiedIncidentAction(action_type="query_deploys", service="worker")
    if "query_logs" in observation.allowed_actions:
        return UnifiedIncidentAction(action_type="query_logs", service=services[0] if services else "database")
    if "query_metrics" in observation.allowed_actions:
        return UnifiedIncidentAction(action_type="query_metrics", service=services[0] if services else "database", metric="cpu")
    action_type = observation.allowed_actions[0]
    return UnifiedIncidentAction(**_default_action_for_type(action_type, observation))


def get_model_action(client: OpenAI | None, observation: UnifiedIncidentObservation) -> tuple[UnifiedIncidentAction, str | None]:
    if client is None:
        return build_fallback_action(observation), "model_unavailable"
    try:
        parsed = parse_action(request_action(client, observation), observation)
        if parsed is not None:
            return parsed, None
    except Exception:
        pass
    return build_fallback_action(observation), "fallback_used"


def run_scenario(client: OpenAI | None, scenario_id: str) -> dict[str, Any]:
    with UnifiedIncidentEnv(base_url=ENV_BASE_URL).sync() as env:
        observation = env.reset(scenario_id=scenario_id).observation
        rewards: list[float] = []
        step = 0
        log_start(task=scenario_id, env=ENV_NAME, model=MODEL_NAME)
        while not observation.done:
            step += 1
            action, error = get_model_action(client, observation)
            result = env.step(action)
            observation = result.observation
            rewards.append(float(result.reward))
            log_step(
                step=step,
                action=json.dumps(action.model_dump(exclude_none=True), separators=(",", ":")),
                reward=float(result.reward),
                done=bool(result.done),
                error=error or observation.failure_type,
            )
        log_end(
            success=bool(observation.done and observation.incident_resolved),
            steps=step,
            score=observation.final_score,
            rewards=rewards,
        )
        return {
            "success": bool(observation.done and observation.incident_resolved),
            "score": observation.final_score,
            "steps": step,
            "rewards": rewards,
        }


def main() -> None:
    client = create_client()
    for scenario_id in SCENARIOS:
        run_scenario(client, scenario_id)


if __name__ == "__main__":
    main()
