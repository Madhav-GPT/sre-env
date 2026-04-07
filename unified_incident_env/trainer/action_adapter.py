"""Strict and lenient action parsers for training and eval."""

from __future__ import annotations

import json
from typing import Any

from ..models import ActionType, UnifiedIncidentAction
from .types import ParseResult

_ALLOWED_KEYS = {
    "action_type",
    "service",
    "metric",
    "vulnerability_type",
    "patch_id",
    "postmortem",
}
_KNOWN_ACTIONS: set[str] = {
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


def _extract_json_text(raw_text: str) -> str:
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
        text = text[start : end + 1]
    return text.strip()


def _compact_action(action: UnifiedIncidentAction) -> dict[str, Any]:
    payload = action.model_dump(exclude_none=True)
    if payload.get("metadata") == {}:
        payload.pop("metadata", None)
    return payload


class StrictActionParser:
    """Exact parser for judge-style evaluation."""

    def parse(self, raw_text: str) -> ParseResult:
        bare = raw_text.strip().strip('"').strip("'")
        if bare in {"inspect_code", "verify_security_fix", "submit_security_fix"}:
            action = UnifiedIncidentAction(action_type=bare)
            return ParseResult(
                parse_status="repaired",
                cleaned_action=_compact_action(action),
                repair_labels=["bare_action_wrapped"],
            )

        try:
            data = json.loads(_extract_json_text(raw_text))
        except Exception as exc:
            return ParseResult(parse_status="invalid_json", error=type(exc).__name__)

        if not isinstance(data, dict):
            return ParseResult(parse_status="invalid_action", error="root must be object")

        repaired_labels: list[str] = []
        cleaned: dict[str, Any] = {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
        repaired = cleaned != data
        if repaired:
            repaired_labels.append("extra_keys_stripped")

        if "action_type" not in cleaned and isinstance(data.get("action"), str):
            if data["action"] in _KNOWN_ACTIONS:
                cleaned["action_type"] = data["action"]
                repaired = True
                repaired_labels.append("action_alias_normalized")

        if (
            "vulnerability_type" not in cleaned
            and isinstance(data.get("vulnerability"), str)
        ):
            cleaned["vulnerability_type"] = data["vulnerability"]
            repaired = True
            repaired_labels.append("vulnerability_alias_normalized")

        metrics_value = data.get("metrics")
        if "metric" not in cleaned and isinstance(metrics_value, list) and len(metrics_value) == 1:
            cleaned["metric"] = metrics_value[0]
            repaired = True
            repaired_labels.append("metric_list_normalized")

        if "metrics" in data and (
            not isinstance(metrics_value, list) or len(metrics_value) != 1
        ):
            return ParseResult(
                parse_status="invalid_action",
                error="metrics alias is ambiguous",
                repair_labels=repaired_labels,
            )

        try:
            action = UnifiedIncidentAction(**cleaned)
        except Exception as exc:
            return ParseResult(
                parse_status="invalid_action",
                error=str(exc),
                repair_labels=repaired_labels,
            )

        return ParseResult(
            parse_status="repaired" if repaired else "ok",
            cleaned_action=_compact_action(action),
            repair_labels=repaired_labels,
        )


class LenientActionAdapter:
    """Training-time parser that repairs small schema mistakes only."""

    def parse(self, raw_text: str) -> ParseResult:
        bare = raw_text.strip().strip('"').strip("'")
        if bare in _KNOWN_ACTIONS:
            try:
                action = UnifiedIncidentAction(action_type=bare)
            except Exception as exc:
                return ParseResult(
                    parse_status="invalid_action",
                    error=str(exc),
                    repair_labels=["bare_action_wrapped"],
                )
            return ParseResult(
                parse_status="repaired",
                cleaned_action=_compact_action(action),
                repair_labels=["bare_action_wrapped"],
            )

        try:
            data = json.loads(_extract_json_text(raw_text))
        except Exception as exc:
            return ParseResult(parse_status="invalid_json", error=type(exc).__name__)

        if not isinstance(data, dict):
            return ParseResult(parse_status="invalid_action", error="root must be object")

        repaired_labels: list[str] = []
        cleaned: dict[str, Any] = {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
        repaired = cleaned != data
        if repaired:
            repaired_labels.append("extra_keys_stripped")

        if "action_type" not in cleaned and isinstance(data.get("action"), str):
            if data["action"] in _KNOWN_ACTIONS:
                cleaned["action_type"] = data["action"]
                repaired = True
                repaired_labels.append("action_alias_normalized")

        if (
            "vulnerability_type" not in cleaned
            and isinstance(data.get("vulnerability"), str)
        ):
            cleaned["vulnerability_type"] = data["vulnerability"]
            repaired = True
            repaired_labels.append("vulnerability_alias_normalized")

        metrics_value = data.get("metrics")
        if "metric" not in cleaned and isinstance(metrics_value, list) and len(metrics_value) == 1:
            cleaned["metric"] = metrics_value[0]
            repaired = True
            repaired_labels.append("metric_list_normalized")

        if "metrics" in data and (
            not isinstance(metrics_value, list) or len(metrics_value) != 1
        ):
            return ParseResult(
                parse_status="invalid_action",
                error="metrics alias is ambiguous",
                repair_labels=repaired_labels,
            )

        try:
            action = UnifiedIncidentAction(**cleaned)
        except Exception as exc:
            return ParseResult(
                parse_status="invalid_action",
                error=str(exc),
                repair_labels=repaired_labels,
            )

        return ParseResult(
            parse_status="repaired" if repaired else "ok",
            cleaned_action=_compact_action(action),
            repair_labels=repaired_labels,
        )
