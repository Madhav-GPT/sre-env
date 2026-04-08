#!/usr/bin/env python3
"""Submission inference script with validator-compatible stdout logs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from unified_incident_env.client import UnifiedIncidentEnv
from unified_incident_env.models import (
    PostmortemPayload,
    SecurityContext,
    UnifiedIncidentAction,
    UnifiedIncidentObservation,
)
from unified_incident_env.server.challenge import SCENARIOS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or UnifiedIncidentEnv.DEFAULT_BASE_URL
ENV_NAME = "unified-incident-env"
MAX_TOKENS = 220

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "judge").strip().lower()
POLICY_CARD_WORD_BUDGET_COMPACT = int(os.getenv("POLICY_CARD_WORD_BUDGET_COMPACT", "60"))

POLICY_CARD_RULES = [
    "Return JSON only.",
    "Use action_type.",
    "Use only allowed actions.",
    "No explanation text.",
]
STAGE_GOALS = {
    "diagnosis": "find the most relevant next investigation step",
    "root_cause_analysis": "confirm the root-cause evidence and avoid unnecessary recovery",
    "security_subquest": "complete the security fix before infrastructure recovery",
    "remediation": "recover services in the correct order",
    "verification": "verify that recovery and security remediation are complete",
    "postmortem": "submit the final incident summary",
    "done": "complete the benchmark",
}

ACTION_KEYS = {
    "action_type",
    "service",
    "metric",
    "vulnerability_type",
    "patch_id",
    "postmortem",
}
KNOWN_ACTIONS = {
    "query_logs",
    "query_metrics",
    "query_dependencies",
    "restart_service",
    "rollback_deploy",
    "inspect_code",
    "classify_vulnerability",
    "apply_patch",
    "verify_security_fix",
    "submit_security_fix",
    "submit_postmortem",
}
LOCAL_ENDPOINT_MARKERS = ("127.0.0.1", "localhost")
SERVICE_PRIORITY = ("database", "cache", "api-gateway", "worker")
VULNERABILITY_KEYWORDS = {
    "sql_injection": ("sql injection", "sqli", "query", "parameter", "login"),
    "broken_access_control": ("access control", "authorization", "admin", "role", "permission"),
    "command_injection": ("command injection", "shell", "subprocess", "filename", "worker"),
}
PATCH_KEYWORDS = {
    "sql_injection": ("parameter", "prepared", "query"),
    "broken_access_control": ("admin", "role", "authoriz"),
    "command_injection": ("avoid_shell", "argv", "shell", "subprocess"),
}

SYSTEM_PROMPT = """You are solving a deterministic incident-response benchmark.

Return exactly one JSON object and nothing else.

Rules:
- Choose only from the allowed action types shown in the user message.
- Use only the required fields for the chosen action.
- Do not include explanation text.
- Do not include markdown.
- Do not include code fences.
- Do not repeat an action that already failed or made no progress.
- If patching is required, use only one of the listed patch IDs.
"""

USER_PROMPT_TEMPLATE = """Current stage: {stage}

Current goal: {goal}

Allowed actions:
{allowed_actions_block}

Required fields:
{required_fields_block}

{patch_ids_block}{transition_block}{negative_reward_block}{loop_warning_block}Current environment state:
{state_block}

Valid example:
{valid_example}

Return exactly one JSON object.
"""


@dataclass
class PolicyNote:
    stage: str
    failure_type: str
    mistake: str
    correction: str
    valid_example: dict[str, Any]
    action_family: str | None = None


@dataclass
class PolicyCardState:
    schema_notes: list[PolicyNote] = field(default_factory=list)
    failure_notes: list[PolicyNote] = field(default_factory=list)
    recovery_notes: list[PolicyNote] = field(default_factory=list)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def action_to_log_string(action: UnifiedIncidentAction) -> str:
    return json.dumps(
        action.model_dump(exclude_none=True, exclude={"metadata"}),
        separators=(",", ":"),
    )


def create_client() -> OpenAI | None:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=45.0)
    except Exception:
        return None


def _inference_mode() -> str:
    return "small" if os.getenv("INFERENCE_MODE", INFERENCE_MODE).strip().lower() == "small" else "judge"


def _is_local_ollama() -> bool:
    return any(marker in API_BASE_URL for marker in LOCAL_ENDPOINT_MARKERS)


def _extract_json_candidate(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def parse_action(
    raw: str,
    observation: UnifiedIncidentObservation,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> UnifiedIncidentAction | None:
    stage_allowed_actions = _narrow_allowed_actions(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )
    text = raw.strip()
    if not text:
        return None

    bare = text.strip().strip('"').strip("'")
    if bare in stage_allowed_actions and bare in KNOWN_ACTIONS:
        fields = observation.required_fields_by_action.get(bare, [])
        if not fields:
            return UnifiedIncidentAction(action_type=bare)
        example = observation.valid_action_example or {}
        if example.get("action_type") == bare:
            try:
                return UnifiedIncidentAction(**example)
            except Exception:
                return None
        return None

    try:
        payload = json.loads(_extract_json_candidate(text))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    cleaned = {key: value for key, value in payload.items() if key in ACTION_KEYS}
    if "action_type" not in cleaned and isinstance(payload.get("action"), str):
        cleaned["action_type"] = payload["action"]
    if "vulnerability_type" not in cleaned and isinstance(payload.get("vulnerability"), str):
        cleaned["vulnerability_type"] = payload["vulnerability"]
    metrics_value = payload.get("metrics")
    if "metric" not in cleaned and isinstance(metrics_value, list) and len(metrics_value) == 1:
        cleaned["metric"] = metrics_value[0]

    action_type = cleaned.get("action_type")
    if action_type not in stage_allowed_actions:
        return None

    try:
        return UnifiedIncidentAction(**cleaned)
    except Exception:
        return None


def choose_investigation_service(observation: UnifiedIncidentObservation) -> str:
    critical_alerts = [
        alert.service for alert in observation.active_alerts if alert.severity == "critical"
    ]
    if critical_alerts:
        return critical_alerts[0]
    for service in SERVICE_PRIORITY:
        health = observation.service_health.get(service)
        if health and health.status == "crashed":
            return service
    for service in SERVICE_PRIORITY:
        health = observation.service_health.get(service)
        if health and health.status == "degraded":
            return service
    return "api-gateway"


def choose_recovery_service(observation: UnifiedIncidentObservation) -> str:
    for service in SERVICE_PRIORITY:
        health = observation.service_health.get(service)
        if health and health.status == "crashed":
            return service
    for service in SERVICE_PRIORITY:
        health = observation.service_health.get(service)
        if health and health.status == "degraded":
            return service
    return "api-gateway"


def infer_vulnerability(observation: UnifiedIncidentObservation, history: list[dict[str, Any]]) -> str:
    text_parts = [
        observation.prompt_text,
        observation.tool_output or "",
        observation.security_unlock_reason or "",
        observation.last_action_result,
        observation.why_failed or "",
    ]
    text_parts.extend(str(item.get("result", "")) for item in history[-4:])
    haystack = " ".join(text_parts).lower()
    best = "sql_injection"
    best_score = -1
    for vulnerability, keywords in VULNERABILITY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > best_score:
            best = vulnerability
            best_score = score
    return best


def extract_patch_options(observation: UnifiedIncidentObservation) -> list[str]:
    sources = [observation.tool_output or "", observation.prompt_text]
    for source in sources:
        match = re.search(r"Patch options:\s*([^\n]+)", source)
        if not match:
            continue
        return [option.strip() for option in match.group(1).split(",") if option.strip()]
    return []


def _allowed_patch_ids(observation: UnifiedIncidentObservation) -> list[str]:
    options = extract_patch_options(observation)
    if not options:
        options = ["parameterized_query", "enforce_admin_role", "avoid_shell"]
    
    # If vulnerability is already classified, filter options to matching family
    vuln = observation.security_context.selected_vulnerability
    if vuln:
        keywords = PATCH_KEYWORDS.get(vuln, [])
        filtered = [
            opt for opt in options 
            if any(k in opt.lower() for k in keywords)
        ]
        if filtered:
            return filtered
            
    return options


def _stage_hint(
    observation: UnifiedIncidentObservation,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    hard = _hard_transition_state(
        scenario_id=scenario_id,
        observation=observation,
        history=history or [],
    )
    if hard["next_required_action"] is not None:
        return hard["next_required_action"]
    if hard["next_required_action_family"] is not None:
        return f"Next required action family: {hard['next_required_action_family']}."
    stage = observation.workflow_stage
    if stage == "diagnosis":
        return "Find the root cause with investigation before moving to security or recovery."
    if stage == "root_cause_analysis":
        return "Confirm the root cause and avoid broad extra queries."
    if stage == "security_subquest":
        return "Solve the security subquest with the next security action."
    if stage == "remediation":
        return "Recover the system with the allowed remediation action."
    if stage == "verification":
        return "Verify the fix before submitting the security fix."
    if stage == "postmortem":
        return "Submit the postmortem after the incident is resolved."
    return "Follow the current stage goal and allowed actions."


def _stop_investigating_hint(
    observation: UnifiedIncidentObservation,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    hard = _hard_transition_state(
        scenario_id=scenario_id,
        observation=observation,
        history=history or [],
    )
    if hard["stop_investigating"]:
        return hard["stop_message"]
    if observation.loop_warning:
        return "Stop repeating the same no-progress action; choose a different allowed action family."
    if observation.workflow_stage == "root_cause_analysis":
        return "Avoid broad investigation; confirm the root cause or move to the next stage."
    if observation.workflow_stage in {"security_subquest", "remediation", "verification", "postmortem"}:
        return "Avoid extra query_* investigation actions unless required by the current stage."
    return None


def choose_patch_id(observation: UnifiedIncidentObservation, history: list[dict[str, Any]]) -> str:
    options = extract_patch_options(observation)
    vulnerability = infer_vulnerability(observation, history)
    keywords = PATCH_KEYWORDS[vulnerability]
    for option in options:
        lowered = option.lower()
        if any(keyword in lowered for keyword in keywords):
            return option
    if options:
        return options[0]
    defaults = {
        "sql_injection": "parameterized_query",
        "broken_access_control": "enforce_admin_role",
        "command_injection": "avoid_shell",
    }
    return defaults[vulnerability]


def _timeline_entry(action: UnifiedIncidentAction) -> str:
    if action.action_type in {"query_logs", "query_dependencies"} and action.service:
        return f"{action.action_type} {action.service}"
    if action.action_type == "query_metrics" and action.service and action.metric:
        return f"query_metrics {action.service}.{action.metric}"
    if action.action_type in {"restart_service", "rollback_deploy"} and action.service:
        return f"{action.action_type} {action.service}"
    if action.action_type == "classify_vulnerability" and action.vulnerability_type:
        return f"classify_vulnerability {action.vulnerability_type}"
    if action.action_type == "apply_patch" and action.patch_id:
        return f"apply_patch {action.patch_id}"
    return action.action_type


def _action_family(action_type: str | None) -> str | None:
    if action_type in {"query_logs", "query_metrics", "query_dependencies"}:
        return "investigate"
    if action_type in {
        "inspect_code",
        "classify_vulnerability",
        "apply_patch",
        "verify_security_fix",
        "submit_security_fix",
    }:
        return "security"
    if action_type in {"restart_service", "rollback_deploy"}:
        return "recovery"
    if action_type == "submit_postmortem":
        return "postmortem"
    return None


def build_postmortem(
    observation: UnifiedIncidentObservation,
    history: list[dict[str, Any]],
) -> PostmortemPayload:
    vulnerability = infer_vulnerability(observation, history)
    selected_patch = observation.security_context.selected_patch

    root_cause_map = {
        "sql_injection": "SQL injection crashed the database and caused gateway errors.",
        "broken_access_control": "Broken access control on an admin path caused cache abuse and database degradation.",
        "command_injection": "Command injection in the worker poisoned downstream services after a bad deploy.",
    }
    attack_vector_map = {
        "sql_injection": "Unsanitized login input abused the SQL query path.",
        "broken_access_control": "Missing admin authorization exposed an internal cache-management route.",
        "command_injection": "Unsafe shell command construction allowed attacker-controlled filenames to execute commands.",
    }
    prevention_map = {
        "sql_injection": ["Parameterized queries", "Database abuse alerting"],
        "broken_access_control": ["Admin role enforcement", "Authorization tests"],
        "command_injection": ["Avoid shell invocation", "Safer deploy validation"],
    }

    timeline = [_timeline_entry(item["action"]) for item in history if "action" in item]
    remediation_steps = []
    if selected_patch:
        remediation_steps.append(selected_patch.replace("_", " "))
    remediation_steps.extend(
        item["action"].service.replace("-", " ")
        for item in history
        if "action" in item
        and item["action"].action_type in {"restart_service", "rollback_deploy"}
        and item["action"].service
    )

    return PostmortemPayload(
        root_cause=root_cause_map[vulnerability],
        attack_vector=attack_vector_map[vulnerability],
        timeline=timeline[-6:],
        remediation_steps=remediation_steps[:4],
        prevention_steps=prevention_map[vulnerability],
    )


def build_fallback_action(
    observation: UnifiedIncidentObservation,
    history: list[dict[str, Any]],
    *,
    scenario_id: str | None = None,
) -> UnifiedIncidentAction:
    hard = _hard_transition_state(
        scenario_id=scenario_id,
        observation=observation,
        history=history,
    )
    example = observation.valid_action_example or {}
    last_action = (
        history[-1]["action"].model_dump(exclude_none=True, exclude={"metadata"})
        if history and "action" in history[-1]
        else None
    )
    narrowed_allowed_actions = _narrow_allowed_actions(
        observation,
        scenario_id=scenario_id,
        history=history,
    )
    if example.get("action_type") in narrowed_allowed_actions and example != last_action:
        try:
            return UnifiedIncidentAction(**example)
        except Exception:
            pass

    stage = observation.workflow_stage
    security: SecurityContext = observation.security_context

    if stage in {"diagnosis", "root_cause_analysis"}:
        if hard["needs_unlock_bridge"]:
            return UnifiedIncidentAction(
                action_type="query_dependencies",
                service="api-gateway",
            )
        if stage == "root_cause_analysis" and "query_dependencies" in observation.allowed_actions:
            return UnifiedIncidentAction(
                action_type="query_dependencies",
                service="api-gateway",
            )
        if "query_logs" in observation.allowed_actions:
            return UnifiedIncidentAction(
                action_type="query_logs",
                service=choose_investigation_service(observation),
            )
        if "query_dependencies" in observation.allowed_actions:
            return UnifiedIncidentAction(
                action_type="query_dependencies",
                service=choose_investigation_service(observation),
            )
        return UnifiedIncidentAction(
            action_type="query_metrics",
            service=choose_investigation_service(observation),
            metric="cpu",
        )

    if stage == "security_subquest":
        if not security.code_visible:
            return UnifiedIncidentAction(action_type="inspect_code")
        if security.selected_vulnerability is None:
            return UnifiedIncidentAction(
                action_type="classify_vulnerability",
                vulnerability_type=infer_vulnerability(observation, history),
            )
        if security.selected_patch is None:
            return UnifiedIncidentAction(
                action_type="apply_patch",
                patch_id=choose_patch_id(observation, history),
            )
        if security.exploit_blocked is not True or security.functionality_preserved is not True:
            return UnifiedIncidentAction(action_type="verify_security_fix")
        return UnifiedIncidentAction(action_type="submit_security_fix")

    if stage in {"remediation", "verification"}:
        if hard["force_worker_rollback"]:
            return UnifiedIncidentAction(action_type="rollback_deploy", service="worker")
        worker = observation.service_health.get("worker")
        if (
            "rollback_deploy" in observation.allowed_actions
            and worker is not None
            and worker.status != "healthy"
        ):
            return UnifiedIncidentAction(action_type="rollback_deploy", service="worker")
        return UnifiedIncidentAction(
            action_type="restart_service",
            service=choose_recovery_service(observation),
        )

    return UnifiedIncidentAction(
        action_type="submit_postmortem",
        postmortem=build_postmortem(observation, history),
    )


def build_compact_policy_card(
    observation: UnifiedIncidentObservation,
    state: PolicyCardState,
    history: list[dict[str, Any]] | None = None,
    *,
    scenario_id: str | None = None,
) -> str:
    """Brutally small policy card for weak backends."""
    if history is None:
        history = []
    stage_allowed_actions = _narrow_allowed_actions(
        observation,
        scenario_id=scenario_id,
        history=history,
    )
    lines = [
        f"STAGE: {observation.workflow_stage}",
        f"GOAL: {STAGE_GOALS.get(observation.workflow_stage, 'Pick one valid action.')}",
        f"HINT: {_stage_hint(observation, scenario_id=scenario_id, history=history)}",
        f"ALLOWED: {', '.join(stage_allowed_actions)}",
    ]
    stop_hint = _stop_investigating_hint(
        observation,
        scenario_id=scenario_id,
        history=history,
    )
    if stop_hint:
        lines.append(f"STOP_INVESTIGATING: {stop_hint}")
    if observation.loop_warning:
        lines.append("LESSON: Do not repeat the same no-progress action.")
    elif state.failure_notes:
        lines.append(f"LESSON: {state.failure_notes[-1].correction}")
    example = observation.valid_action_example or {"action_type": stage_allowed_actions[0]}
    lines.append(f"EXAMPLE: {json.dumps(example, separators=(',', ':'))}")
    if "apply_patch" in stage_allowed_actions:
        lines.append(f"PATCH_IDS: {', '.join(_allowed_patch_ids(observation))}")
    lines.append("Return exactly one JSON object.")
    return _limit_words("\n".join(lines), max_words=POLICY_CARD_WORD_BUDGET_COMPACT)


def build_policy_card(
    observation: UnifiedIncidentObservation,
    state: PolicyCardState,
    history: list[dict[str, Any]] | None = None,
    *,
    scenario_id: str | None = None,
) -> str:
    """Always use compact mode for small-model inference."""
    return build_compact_policy_card(
        observation,
        state,
        history or [],
        scenario_id=scenario_id,
    )


def update_policy_card(
    state: PolicyCardState,
    *,
    before: UnifiedIncidentObservation,
    action: UnifiedIncidentAction,
    after: UnifiedIncidentObservation,
    model_error: str | None,
) -> None:
    if model_error == "invalid_model_output":
        state.schema_notes.append(
            PolicyNote(
                stage=before.workflow_stage,
                failure_type="invalid_model_output",
                mistake="The previous response was not one valid JSON action object.",
                correction="Return exactly one valid JSON action using only allowed actions.",
                valid_example=before.valid_action_example or {"action_type": before.allowed_actions[0]},
                action_family=_action_family((before.valid_action_example or {}).get("action_type")),
            )
        )
        state.schema_notes = state.schema_notes[-4:]

    if after.failure_type and after.why_failed:
        example = after.valid_action_example or before.valid_action_example or {"action_type": before.allowed_actions[0]}
        family = after.best_recovery_action_family or _action_family(example.get("action_type"))
        correction = (
            f"If this happens again, prefer {family} actions."
            if family
            else "Follow the current stage example and allowed actions."
        )
        
        state.failure_notes.append(
            PolicyNote(
                stage=before.workflow_stage,
                failure_type=after.failure_type,
                mistake=after.why_failed,
                correction=correction,
                valid_example=example,
                action_family=family,
            )
        )
        state.failure_notes = state.failure_notes[-4:]

    if after.reward > 0 and after.failure_type is None:
        state.recovery_notes.append(
            PolicyNote(
                stage=before.workflow_stage,
                failure_type="successful_step",
                mistake="A weaker choice would likely have lost progress.",
                correction=f"This stage can progress with {_timeline_entry(action)}.",
                valid_example=action.model_dump(exclude_none=True, exclude={"metadata"}),
                action_family=_action_family(action.action_type),
            )
        )
        state.recovery_notes = state.recovery_notes[-4:]


def _build_required_fields_block(
    required_fields_by_action: dict[str, list[str]],
    allowed_actions: list[str],
) -> str:
    lines = []
    for action in allowed_actions:
        fields = required_fields_by_action.get(action, [])
        if fields:
            lines.append(f"- {action} -> {', '.join(fields)}")
        else:
            lines.append(f"- {action} -> none")
    return "\n".join(lines) or "- none"


def _build_patch_ids_block(patch_ids: list[str]) -> str:
    if not patch_ids:
        return ""
    lines = ["Available patch IDs:"]
    lines.extend(f"- {patch_id}" for patch_id in patch_ids)
    lines.append("")
    return "\n".join(lines)


def _build_transition_block(transition_hint: str | None) -> str:
    if not transition_hint:
        return ""
    return f"Important transition hint:\n- {transition_hint}\n\n"


def _build_negative_reward_block(correction_hint: str | None) -> str:
    if not correction_hint:
        return ""
    return f"Previous action correction:\n- {correction_hint}\n\n"


def _build_loop_warning_block(loop_warning: str | None) -> str:
    if not loop_warning:
        return ""
    return f"Loop warning:\n- {loop_warning}\n\n"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return str(value).lower()


def _render_tool_output(observation: UnifiedIncidentObservation) -> str:
    if not observation.tool_output:
        return ""

    if observation.workflow_stage in {"security_subquest", "verification"}:
        lines = [line.rstrip() for line in observation.tool_output.splitlines() if line.strip()]
        return "\n".join(lines[:6])

    return observation.tool_output.splitlines()[0]


def _build_state_block(observation: UnifiedIncidentObservation) -> str:
    lines: list[str] = []
    if observation.active_alerts:
        lines.append("Active alerts:")
        for alert in observation.active_alerts[:3]:
            lines.append(f"- {alert.service}: {alert.severity} - {alert.message}")

    lines.append(f"Final score: {observation.final_score:.4f}")

    if observation.last_action_result:
        lines.append(f"Last action result: {observation.last_action_result}")

    if observation.tool_output:
        rendered_tool_output = _render_tool_output(observation)
        if "\n" in rendered_tool_output:
            lines.append("Tool output:")
            lines.extend(rendered_tool_output.splitlines())
        else:
            lines.append(f"Tool output: {rendered_tool_output}")

    security = observation.security_context
    if observation.workflow_stage in {"security_subquest", "verification"}:
        lines.append(
            "Security status: "
            f"code visible = {str(security.code_visible).lower()}, "
            f"vulnerability classified = {str(security.selected_vulnerability is not None).lower()}, "
            f"patch applied = {str(security.selected_patch is not None).lower()}, "
            f"exploit blocked = {_bool_text(security.exploit_blocked)}, "
            f"functionality preserved = {_bool_text(security.functionality_preserved)}"
        )

    if observation.security_unlock_reason:
        lines.append(f"Security unlock reason: {observation.security_unlock_reason}")

    if observation.blocked_until_security_complete:
        lines.append("Recovery gate: security must be completed before recovery.")

    return "\n".join(lines) or "- none"


def _extract_policy_hint(policy_card: str) -> str | None:
    for prefix in ("LESSON:", "STOP_INVESTIGATING:"):
        for line in policy_card.splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    return None


def _user_prompt_example(
    observation: UnifiedIncidentObservation,
    allowed_actions: list[str],
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    example = observation.valid_action_example or {}
    if example.get("action_type") in allowed_actions:
        return example
    fallback = build_fallback_action(
        observation,
        history or [],
        scenario_id=scenario_id,
    )
    return fallback.model_dump(exclude_none=True, exclude={"metadata"})


def build_user_prompt(
    observation: UnifiedIncidentObservation,
    policy_card: str,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    stage_allowed_actions = _narrow_allowed_actions(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )
    required_fields = observation.required_fields_by_action or {
        action: []
        for action in stage_allowed_actions
    }
    transition_hint = _stop_investigating_hint(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    ) or _stage_hint(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )
    correction_hint = None
    if observation.failure_type and observation.why_failed:
        correction_hint = observation.why_failed
    elif policy_card:
        correction_hint = _extract_policy_hint(policy_card)

    valid_example = _user_prompt_example(
        observation,
        stage_allowed_actions,
        scenario_id=scenario_id,
        history=history,
    )
    return USER_PROMPT_TEMPLATE.format(
        stage=observation.workflow_stage,
        goal=STAGE_GOALS.get(observation.workflow_stage, "take the best next action"),
        allowed_actions_block="\n".join(f"- {action}" for action in stage_allowed_actions) or "- none",
        required_fields_block=_build_required_fields_block(required_fields, stage_allowed_actions),
        patch_ids_block=_build_patch_ids_block(
            _allowed_patch_ids(observation) if "apply_patch" in stage_allowed_actions else []
        ),
        transition_block=_build_transition_block(transition_hint),
        negative_reward_block=_build_negative_reward_block(correction_hint),
        loop_warning_block=_build_loop_warning_block(observation.loop_warning),
        state_block=_build_state_block(observation),
        valid_example=json.dumps(valid_example, separators=(",", ":")),
    )


def _build_tool_schema(
    observation: UnifiedIncidentObservation,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    allowed_actions = _narrow_allowed_actions(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )
    properties: dict[str, Any] = {
        "action_type": {"type": "string", "enum": allowed_actions},
    }
    if any(action in allowed_actions for action in {"query_logs", "query_metrics", "query_dependencies", "restart_service", "rollback_deploy"}):
        properties["service"] = {
            "type": "string",
            "enum": sorted(observation.service_health.keys()),
        }
    if "query_metrics" in allowed_actions:
        properties["metric"] = {
            "type": "string",
            "enum": ["cpu", "memory", "latency", "error_rate", "throughput"],
        }
    if "classify_vulnerability" in allowed_actions:
        properties["vulnerability_type"] = {
            "type": "string",
            "enum": ["sql_injection", "broken_access_control", "command_injection"],
        }
    if "apply_patch" in allowed_actions:
        properties["patch_id"] = {
            "type": "string",
            "enum": _allowed_patch_ids(observation),
        }
    if "submit_postmortem" in allowed_actions:
        properties["postmortem"] = {"type": "object"}
    required = ["action_type"]
    example = observation.valid_action_example or {}
    for field in ("service", "metric", "vulnerability_type", "patch_id", "postmortem"):
        if field in properties and field in example:
            required.append(field)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _extract_completion_text(completion) -> str:
    message = completion.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        function = getattr(tool_calls[0], "function", None)
        if function is not None and getattr(function, "arguments", None):
            return function.arguments
    return (message.content or "").strip()


def _request_action_completion(
    client: OpenAI,
    observation: UnifiedIncidentObservation,
    user_prompt: str,
    *,
    temperature: float,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    import time
    max_retries = 3
    last_exc = None
    
    schema = _build_tool_schema(
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )

    for attempt in range(max_retries):
        try:
            create_kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }
            
            if _is_local_ollama():
                create_kwargs["extra_body"] = {"format": schema}
                completion = client.chat.completions.create(**create_kwargs)
                return _extract_completion_text(completion)

            try:
                # Try tool calling first
                completion = client.chat.completions.create(
                    **create_kwargs,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "emit_action",
                                "description": "Emit exactly one environment action.",
                                "parameters": schema,
                            },
                        }
                    ],
                    tool_choice={"type": "function", "function": {"name": "emit_action"}},
                )
                return _extract_completion_text(completion)
            except Exception:
                # Fallback to JSON mode
                completion = client.chat.completions.create(
                    **create_kwargs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "unified_incident_action",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                return _extract_completion_text(completion)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise last_exc
    return ""


def attempt_repair(
    client: OpenAI,
    observation: UnifiedIncidentObservation,
    raw_output: str,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> UnifiedIncidentAction | None:
    example = observation.valid_action_example or {
        "action_type": _narrow_allowed_actions(
            observation,
            scenario_id=scenario_id,
            history=history or [],
        )[0]
    }
    repair_prompt = (
        "Your previous response was invalid.\n"
        "Return exactly one valid JSON object.\n"
        "No explanation.\n"
        f"Example: {json.dumps(example, separators=(',', ':'))}\n"
        f"Previous response: {raw_output}"
    )
    try:
        repaired = _request_action_completion(
            client,
            observation,
            repair_prompt,
            temperature=0.0,
            scenario_id=scenario_id,
            history=history or [],
        )
    except Exception:
        return None
    return parse_action(
        repaired,
        observation,
        scenario_id=scenario_id,
        history=history or [],
    )


def get_model_action(
    client: OpenAI | None,
    observation: UnifiedIncidentObservation,
    history: list[dict[str, Any]],
    policy_state: PolicyCardState,
    *,
    scenario_id: str | None = None,
) -> tuple[UnifiedIncidentAction, str | None, bool, bool]:
    fallback = build_fallback_action(observation, history, scenario_id=scenario_id)
    mode = _inference_mode()
    if client is None:
        return fallback, "model_unavailable", False, True

    try:
        policy_card = (
            build_policy_card(
                observation,
                policy_state,
                history,
                scenario_id=scenario_id,
            )
            if mode == "small"
            else ""
        )
        raw = _request_action_completion(
            client,
            observation,
            build_user_prompt(
                observation,
                policy_card,
                scenario_id=scenario_id,
                history=history,
            ),
            temperature=0.0,
            scenario_id=scenario_id,
            history=history,
        )
    except Exception:
        return fallback, "model_request_failed", False, True

    parsed = parse_action(
        raw,
        observation,
        scenario_id=scenario_id,
        history=history,
    )
    if parsed is None:
        repaired = attempt_repair(
            client,
            observation,
            raw,
            scenario_id=scenario_id,
            history=history,
        )
        if repaired is not None:
            return repaired, "repair_retry_used", True, False
        return fallback, "invalid_model_output", True, True
    return parsed, None, False, False


def run_scenario(client: OpenAI | None, scenario_id: str) -> dict[str, Any]:
    import time

    started = time.perf_counter()
    with UnifiedIncidentEnv(base_url=ENV_BASE_URL).sync() as env:
        observation = env.reset(scenario_id=scenario_id).observation
        history: list[dict[str, Any]] = []
        rewards: list[float] = []
        policy_state = PolicyCardState()
        repair_retry_count = 0
        fallback_count = 0
        log_start(task=scenario_id, env=ENV_NAME, model=MODEL_NAME)

        step = 0
        while not observation.done:
            before = observation
            action, error, used_repair_retry, used_fallback = get_model_action(
                client,
                observation,
                history,
                policy_state,
                scenario_id=scenario_id,
            )
            if used_repair_retry:
                repair_retry_count += 1
            if used_fallback:
                fallback_count += 1
            result = env.step(action)
            observation = result.observation
            reward = result.reward or 0.0
            step += 1
            rewards.append(reward)
            history.append(
                {
                    "action": action,
                    "reward": reward,
                    "result": observation.last_action_result,
                    "error": error,
                }
            )
            if _inference_mode() == "small":
                update_policy_card(
                    policy_state,
                    before=before,
                    action=action,
                    after=observation,
                    model_error=error,
                )
            log_step(
                step=step,
                action=action_to_log_string(action),
                reward=reward,
                done=bool(result.done),
                error=error,
            )

        success = bool(
            observation.done
            and observation.incident_resolved
            and observation.security_subquest_status == "completed"
        )
        log_end(
            success=success,
            steps=step,
            score=observation.final_score,
            rewards=rewards,
        )
        return {
            "scenario_id": scenario_id,
            "score": observation.final_score,
            "success": success,
            "steps": step,
            "repair_retry_triggered": repair_retry_count > 0,
            "repair_retry_count": repair_retry_count,
            "fallback_triggered": fallback_count > 0,
            "fallback_count": fallback_count,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }


def main() -> None:
    client = create_client()
    for scenario_id in SCENARIOS:
        run_scenario(client, scenario_id)


def _limit_words(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip() + " ..."


def _narrow_allowed_actions(
    observation: UnifiedIncidentObservation,
    *,
    scenario_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> list[str]:
    allowed_actions = observation.allowed_actions or sorted(KNOWN_ACTIONS)
    hard = _hard_transition_state(
        scenario_id=scenario_id,
        observation=observation,
        history=history or [],
    )
    if hard["force_worker_rollback"] and "rollback_deploy" in allowed_actions:
        return ["rollback_deploy"]
    if hard["needs_unlock_bridge"] and "query_dependencies" in allowed_actions:
        return ["query_dependencies"]
    if hard["security_only"]:
        security_actions = [
            action for action in allowed_actions
            if action in {
                "inspect_code",
                "classify_vulnerability",
                "apply_patch",
                "verify_security_fix",
                "submit_security_fix",
            }
        ]
        if security_actions:
            allowed_actions = security_actions
    if observation.workflow_stage not in {"security_subquest", "verification"}:
        return allowed_actions

    context = observation.security_context
    if not context.code_visible and "inspect_code" in allowed_actions:
        return ["inspect_code"]
    if context.code_visible and context.selected_vulnerability is None and "classify_vulnerability" in allowed_actions:
        return ["classify_vulnerability"]
    if context.selected_vulnerability is not None and context.selected_patch is None and "apply_patch" in allowed_actions:
        return ["apply_patch"]
    if (
        context.selected_patch is not None
        and (context.exploit_blocked is not True or context.functionality_preserved is not True)
        and "verify_security_fix" in allowed_actions
    ):
        return ["verify_security_fix"]
    if (
        context.exploit_blocked is True
        and context.functionality_preserved is True
        and "submit_security_fix" in allowed_actions
    ):
        return ["submit_security_fix"]
    return allowed_actions


def _hard_transition_state(
    *,
    scenario_id: str | None,
    observation: UnifiedIncidentObservation,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    default = {
        "investigation_saturated": False,
        "stop_investigating": False,
        "stop_message": None,
        "next_required_action_family": None,
        "next_required_action": None,
        "needs_unlock_bridge": False,
        "security_only": False,
        "force_worker_rollback": False,
    }
    if scenario_id != "worker_bad_deploy_command_injection":
        return default

    worker_log_queries = sum(
        1
        for item in history
        if item.get("action") is not None
        and item["action"].action_type == "query_logs"
        and item["action"].service == "worker"
    )
    support_queries = sum(
        1
        for item in history
        if item.get("action") is not None
        and (
            (item["action"].action_type == "query_metrics" and item["action"].service in {"worker", "database"})
            or (item["action"].action_type == "query_dependencies" and item["action"].service == "api-gateway")
        )
    )
    investigation_saturated = worker_log_queries >= 1 and (support_queries >= 1 or observation.workflow_stage != "diagnosis")
    security_completed = observation.security_subquest_status == "completed"
    security_unlocked = observation.security_subquest_status != "locked"
    worker_unhealthy = (
        observation.service_health.get("worker") is not None
        and observation.service_health["worker"].status != "healthy"
    )

    if security_completed and worker_unhealthy:
        return {
            **default,
            "investigation_saturated": True,
            "stop_investigating": True,
            "stop_message": "Investigation is complete. The bad worker deploy is still active. Choose rollback_deploy on worker next.",
            "next_required_action_family": "recovery",
            "next_required_action": "Next required action: rollback_deploy on worker.",
            "force_worker_rollback": True,
        }

    if investigation_saturated and not security_unlocked:
        return {
            **default,
            "investigation_saturated": True,
            "stop_investigating": True,
            "stop_message": "You already have enough evidence from worker investigation. Do not query worker logs again. Use query_dependencies on api-gateway to unlock the exploit path.",
            "next_required_action_family": "security",
            "next_required_action": "Next bridge action: query_dependencies on api-gateway, then move to security.",
            "needs_unlock_bridge": True,
        }

    if investigation_saturated and security_unlocked and not security_completed:
        return {
            **default,
            "investigation_saturated": True,
            "stop_investigating": True,
            "stop_message": "Repeated worker investigation is making no progress. Investigation is complete. Choose a security action now.",
            "next_required_action_family": "security",
            "next_required_action": "Current goal: inspect and patch the worker exploit path.",
            "security_only": True,
        }

    if worker_log_queries >= 2 and not security_completed:
        return {
            **default,
            "stop_investigating": True,
            "stop_message": "Repeated worker investigation is making no progress. Choose a different allowed action. Investigation is complete.",
        }

    return default


if __name__ == "__main__":
    main()
