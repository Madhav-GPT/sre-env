"""Prompt builders for training, repair, and strict evaluation."""

from __future__ import annotations

import json
from typing import Any

from ..models import UnifiedIncidentObservation

ALL_ACTIONS = [
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
]

TRAINING_SYSTEM_PROMPT = (
    "You are solving a deterministic incident-response benchmark.\n\n"
    "Return exactly one JSON object and nothing else.\n\n"
    "Rules:\n"
    "- Choose only from the allowed action types shown in the user message.\n"
    "- Use only the required fields for the chosen action.\n"
    "- Do not include explanation text.\n"
    "- Do not include markdown.\n"
    "- Do not include code fences.\n"
    "- Do not repeat an action that already failed or made no progress.\n"
    "- If patching is required, use only one of the listed patch IDs.\n"
)
STAGE_GOALS = {
    "diagnosis": "find the most relevant next investigation step",
    "root_cause_analysis": "confirm the root-cause evidence and avoid unnecessary recovery",
    "security_subquest": "complete the security fix before infrastructure recovery",
    "remediation": "recover services in the correct order",
    "verification": "verify that recovery and security remediation are complete",
    "postmortem": "submit the final incident summary",
    "done": "complete the benchmark",
}

STAGE_ACTIONS = {
    "diagnosis": ["query_logs", "query_metrics", "query_dependencies"],
    "root_cause_analysis": ["query_logs", "query_metrics", "query_dependencies"],
    "security_subquest": [
        "inspect_code",
        "classify_vulnerability",
        "apply_patch",
        "verify_security_fix",
        "submit_security_fix",
    ],
    "remediation": ["restart_service", "rollback_deploy"],
    "verification": ["restart_service", "rollback_deploy"],
    "postmortem": ["submit_postmortem"],
    "done": [],
}

REQUIRED_FIELDS = {
    "query_logs": ["service"],
    "query_metrics": ["service", "metric"],
    "query_dependencies": ["service"],
    "restart_service": ["service"],
    "rollback_deploy": ["service"],
    "inspect_code": [],
    "classify_vulnerability": ["vulnerability_type"],
    "apply_patch": ["patch_id"],
    "verify_security_fix": [],
    "submit_security_fix": [],
    "submit_postmortem": ["postmortem"],
}

USER_PROMPT_TEMPLATE = """Current stage: {stage}

Current goal: {goal}

Allowed actions:
{allowed_actions_block}

Required fields:
{required_fields_block}

{patch_ids_block}{transition_block}{negative_reward_block}Current environment state:
{state_block}

Valid example:
{valid_example}

Return exactly one JSON object.
"""


def build_runtime_request(
    observation: UnifiedIncidentObservation,
    *,
    teacher_action: dict[str, Any] | None = None,
    correction_memory_text: str = "",
    strict: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Build staged prompts and JSON schema guidance for one step."""
    stage = observation.workflow_stage
    allowed_actions = _narrow_allowed_actions(
        observation.allowed_actions or STAGE_ACTIONS.get(stage, ALL_ACTIONS),
        observation,
    )
    required_fields = observation.required_fields_by_action or {
        action: REQUIRED_FIELDS[action] for action in allowed_actions
    }
    example = teacher_action or observation.valid_action_example or _default_example(stage, observation)
    if example.get("action_type") not in allowed_actions:
        example = _default_example(stage, observation)
    user_prompt = _build_user_prompt(
        observation,
        allowed_actions,
        required_fields,
        example=example,
        correction_hint=correction_memory_text.strip() or None,
    )

    response_format = _response_format_for_actions(observation, allowed_actions)
    return TRAINING_SYSTEM_PROMPT, user_prompt, response_format


def build_repair_request(
    observation: UnifiedIncidentObservation,
    *,
    raw_bad_output: str,
    parse_error: str,
    teacher_action: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build one-shot repair prompt after a schema failure."""
    stage = observation.workflow_stage
    example = teacher_action or observation.valid_action_example or _default_example(stage, observation)
    system = (
        "Your previous response was invalid.\n"
        "Return exactly one valid JSON object.\n"
        "No explanation. No extra keys."
    )
    user = (
        f"Stage: {stage}\n"
        f"Error: {parse_error}\n"
        f"Bad output: {raw_bad_output}\n"
        f"Example: {json.dumps(example, separators=(',', ':'))}"
    )
    return system, user


def _default_example(
    stage: str,
    observation: UnifiedIncidentObservation,
) -> dict[str, Any]:
    if stage in {"diagnosis", "root_cause_analysis"}:
        unhealthy = [
            service
            for service, payload in observation.service_health.items()
            if payload.status != "healthy"
        ]
        service = unhealthy[0] if unhealthy else "database"
        if stage == "diagnosis":
            if service == "database":
                return {"action_type": "query_logs", "service": "database"}
            return {"action_type": "query_metrics", "service": service, "metric": "cpu"}
        return {"action_type": "query_dependencies", "service": "api-gateway"}
    if stage == "security_subquest":
        context = observation.security_context
        if not context.code_visible:
            return {"action_type": "inspect_code"}
        if context.selected_vulnerability is None:
            return {
                "action_type": "classify_vulnerability",
                "vulnerability_type": "sql_injection",
            }
        if context.selected_patch is None:
            return {"action_type": "apply_patch", "patch_id": "parameterized_query"}
        if context.exploit_blocked is not True:
            return {"action_type": "verify_security_fix"}
        return {"action_type": "submit_security_fix"}
    if stage in {"remediation", "verification"}:
        unhealthy = [
            service
            for service, payload in observation.service_health.items()
            if payload.status != "healthy"
        ]
        service = unhealthy[0] if unhealthy else "database"
        return {"action_type": "restart_service", "service": service}
    if stage == "postmortem":
        return {
            "action_type": "submit_postmortem",
            "postmortem": {
                "root_cause": "The security issue caused the outage.",
                "attack_vector": "The vulnerable path was abused.",
                "timeline": ["Investigated", "Patched", "Recovered"],
                "remediation_steps": ["Patch", "Restart"],
                "prevention_steps": ["Detection", "Hardening"],
            },
        }
    return {"action_type": "query_logs", "service": "database"}


def _response_format_for_actions(
    observation: UnifiedIncidentObservation,
    allowed_actions: list[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action_type": {
            "type": "string",
            "enum": allowed_actions,
        }
    }
    if any(action in allowed_actions for action in {"query_logs", "query_metrics", "query_dependencies", "restart_service", "rollback_deploy"}):
        services = sorted(observation.service_health.keys())
        properties["service"] = {"type": "string", "enum": services}
    if "query_metrics" in allowed_actions:
        properties["metric"] = {"type": "string", "enum": ["cpu", "memory", "latency", "error_rate", "throughput"]}
    if "classify_vulnerability" in allowed_actions:
        properties["vulnerability_type"] = {
            "type": "string",
            "enum": ["sql_injection", "broken_access_control", "command_injection"],
        }
    if "apply_patch" in allowed_actions:
        patch_options = _patch_options(observation)
        properties["patch_id"] = {
            "type": "string",
            "enum": patch_options or ["parameterized_query", "enforce_admin_role", "avoid_shell"],
        }
    if "submit_postmortem" in allowed_actions:
        properties["postmortem"] = {"type": "object"}

    required = ["action_type"]
    example = observation.valid_action_example or {}
    for field in ("service", "metric", "vulnerability_type", "patch_id", "postmortem"):
        if field in properties and field in example:
            required.append(field)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "unified_incident_action",
            "strict": True,
            "schema": schema,
        },
    }


def _patch_options(observation: UnifiedIncidentObservation) -> list[str]:
    sources = [observation.tool_output or "", observation.prompt_text]
    for source in sources:
        marker = "Patch options:"
        if marker not in source:
            continue
        suffix = source.split(marker, 1)[1].splitlines()[0]
        return [item.strip() for item in suffix.split(",") if item.strip()]
    return []


def _build_required_fields_block(
    required_fields: dict[str, list[str]],
    allowed_actions: list[str],
) -> str:
    lines = []
    for action in allowed_actions:
        fields = required_fields.get(action, [])
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


def _build_transition_block(observation: UnifiedIncidentObservation) -> str:
    transition_hint = _transition_hint(observation)
    if not transition_hint:
        return ""
    return f"Important transition hint:\n- {transition_hint}\n\n"


def _build_negative_reward_block(correction_hint: str | None) -> str:
    if not correction_hint:
        return ""
    compact = correction_hint.splitlines()[0].strip()
    return f"Previous action correction:\n- {compact}\n\n"


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


def _transition_hint(observation: UnifiedIncidentObservation) -> str | None:
    if observation.blocked_until_security_complete:
        return "security must be completed before recovery"
    stage = observation.workflow_stage
    security = observation.security_context
    if stage == "root_cause_analysis":
        return "confirm the root cause with one more relevant investigation step"
    if stage == "security_subquest":
        if not security.code_visible:
            return "inspection must happen before classification or patching"
        if security.selected_vulnerability is None:
            return "classification is required before patching"
        if security.selected_patch is None:
            return "classification is complete; next required action family is patching"
        if security.exploit_blocked is not True or security.functionality_preserved is not True:
            return "patching is complete; verification is required before submission"
        return "security is fixed; submit the security fix before recovery"
    if stage == "remediation":
        return "recover infrastructure only after security is completed"
    if stage == "verification":
        return "verify both security remediation and infrastructure recovery"
    if stage == "postmortem":
        return "submit the final incident summary"
    return None


def _build_user_prompt(
    observation: UnifiedIncidentObservation,
    allowed_actions: list[str],
    required_fields: dict[str, list[str]],
    *,
    example: dict[str, Any],
    correction_hint: str | None = None,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        stage=observation.workflow_stage,
        goal=STAGE_GOALS.get(observation.workflow_stage, "take the best next action"),
        allowed_actions_block="\n".join(f"- {action}" for action in allowed_actions) or "- none",
        required_fields_block=_build_required_fields_block(required_fields, allowed_actions),
        patch_ids_block=_build_patch_ids_block(
            _patch_options(observation) if "apply_patch" in allowed_actions else []
        ),
        transition_block=_build_transition_block(observation),
        negative_reward_block=_build_negative_reward_block(correction_hint),
        state_block=_build_state_block(observation),
        valid_example=json.dumps(example, separators=(",", ":")),
    )


def _narrow_allowed_actions(
    allowed_actions: list[str],
    observation: UnifiedIncidentObservation,
) -> list[str]:
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
