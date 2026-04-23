"""Honest narrow incident-remediation environment core."""

from __future__ import annotations

import json
import uuid
from typing import Any

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import (
    Alert,
    CheckResult,
    ServiceHealth,
    UnifiedIncidentAction,
    UnifiedIncidentObservation,
    UnifiedIncidentState,
)
from .challenge import DEFAULT_SCENARIO_ID, SCENARIOS, get_scenario, scenario_for_difficulty, set_runtime_progress
from .grader import UnifiedIncidentGrader

SERVICE_ORDER = ("api-gateway", "cache", "database", "worker")
ALL_ACTIONS = [
    "query_logs",
    "query_metrics",
    "query_dependencies",
    "query_deploys",
    "rollback_deploy",
    "restart_service",
    "run_check",
    "isolate_service",
    "escalate",
    "submit_hypothesis",
    "declare_resolved",
]
REQUIRED_FIELDS_BY_ACTION: dict[str, list[str]] = {
    "query_logs": ["service"],
    "query_metrics": ["service", "metric"],
    "query_dependencies": ["service"],
    "query_deploys": ["service"],
    "rollback_deploy": ["service"],
    "restart_service": ["service"],
    "run_check": ["check_name"],
    "isolate_service": ["service"],
    "escalate": [],
    "submit_hypothesis": ["hypothesis"],
    "declare_resolved": [],
}
STATUS_VALUES = {
    "healthy": 1.0,
    "degraded": 0.4,
    "crashed": 0.0,
    "isolated": 0.2,
}


class UnifiedIncidentEnvironment(Environment[UnifiedIncidentAction, UnifiedIncidentObservation, UnifiedIncidentState]):
    """A bounded-action incident diagnosis and safe remediation environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._grader = UnifiedIncidentGrader()
        self._episode = self._make_episode(get_scenario(DEFAULT_SCENARIO_ID))
        set_runtime_progress(self._state_dict())

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="unified_incident_env",
            description=(
                "A narrow incident diagnosis and safe remediation environment with bounded actions, "
                "world-state transitions, explicit checks, and effect-based rewards."
            ),
            version="2.0.0",
            author="Daksh Verma",
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> UnifiedIncidentObservation:
        del seed
        scenario_id = kwargs.get("scenario_id")
        difficulty = kwargs.get("difficulty")
        if scenario_id:
            scenario = get_scenario(scenario_id)
        elif difficulty:
            scenario = scenario_for_difficulty(difficulty)
        else:
            scenario = get_scenario(DEFAULT_SCENARIO_ID)
        self._episode = self._make_episode(scenario, episode_id=episode_id)
        set_runtime_progress(self._state_dict())
        return self._build_observation(
            last_action_result="Episode reset.",
            tool_output=None,
            reward=0.0,
            done=False,
        )

    def step(self, action: UnifiedIncidentAction | dict[str, Any], timeout_s: float | None = None, **kwargs: Any) -> UnifiedIncidentObservation:
        del timeout_s, kwargs
        if isinstance(action, dict):
            action = UnifiedIncidentAction(**action)

        if self._episode["done"]:
            return self._build_observation(
                last_action_result="Episode complete. Reset to start another run.",
                tool_output=None,
                reward=0.0,
                done=True,
            )

        self._episode["tick"] += 1
        self._episode["step_count"] += 1
        before_potential = self._incident_health_potential()
        base_step_cost = float(self._episode["scenario"]["reward_config"]["step_cost"])
        penalty = 0.0
        bonus = 0.0
        tool_output: str | None = None
        state_changed = False
        useful_observation = False

        self._episode["failure_type"] = None
        self._episode["why_failed"] = None
        self._episode["loop_warning"] = None

        if action.action_type == "query_logs":
            tool_output = self._query_logs(action.service)
            useful_observation = self._mark_evidence_once(f"logs:{action.service}", tool_output)
            last_action_result = f"Queried logs for {action.service}."
        elif action.action_type == "query_metrics":
            tool_output = self._query_metrics(action.service, action.metric)
            useful_observation = self._mark_evidence_once(f"metrics:{action.service}:{action.metric}", tool_output)
            last_action_result = f"Queried {action.metric} for {action.service}."
        elif action.action_type == "query_dependencies":
            tool_output = self._query_dependencies(action.service)
            useful_observation = self._mark_evidence_once(f"deps:{action.service}", tool_output)
            last_action_result = f"Queried dependencies for {action.service}."
        elif action.action_type == "query_deploys":
            tool_output = self._query_deploys(action.service)
            useful_observation = self._mark_evidence_once(f"deploys:{action.service}", tool_output)
            last_action_result = f"Queried deploy history for {action.service}."
        elif action.action_type == "submit_hypothesis":
            bonus, useful_observation, last_action_result = self._submit_hypothesis(action)
        elif action.action_type == "rollback_deploy":
            state_changed, penalty, last_action_result = self._rollback_deploy(action.service)
        elif action.action_type == "restart_service":
            state_changed, penalty, last_action_result = self._restart_service(action.service)
        elif action.action_type == "isolate_service":
            state_changed, penalty, last_action_result = self._isolate_service(action.service)
        elif action.action_type == "run_check":
            tool_output, useful_observation, last_action_result = self._run_check(action.check_name)
        elif action.action_type == "escalate":
            useful_observation = self._mark_evidence_once(
                f"escalate:{self._episode['tick']}",
                "Escalation note recorded: expert attention requested while keeping the environment state unchanged.",
            )
            last_action_result = "Escalated for human attention."
            tool_output = "Escalation does not fix the incident, but records that expert attention was requested."
        elif action.action_type == "declare_resolved":
            resolved, penalty, bonus, last_action_result = self._declare_resolved()
            state_changed = resolved
        else:
            last_action_result = f"Unsupported action {action.action_type!r}."
            penalty += self._unsafe_penalty()
            self._set_failure("unsupported_action", "That action is not part of this honest narrow environment.")

        self._advance_world()
        self._refresh_alerts()
        self._update_loop_feedback(action, useful_observation or state_changed)
        after_potential = self._incident_health_potential()

        reward = -base_step_cost + (after_potential - before_potential) + bonus - penalty
        if not useful_observation and not state_changed and bonus <= 0.0:
            self._episode["wasteful_ticks"] += 1

        if self._episode["tick"] >= self._episode["max_ticks"] and not self._episode["done"]:
            self._episode["done"] = True
            last_action_result = f"{last_action_result} Tick budget exhausted.".strip()

        self._episode["last_action_result"] = last_action_result
        self._episode["workflow_stage"] = self._workflow_stage()
        self._episode["score_breakdown"] = self._grader.compute_breakdown(self._state_dict(), self._episode["scenario"])
        self._episode["final_score"] = self._episode["score_breakdown"]["final_score"]
        self._episode["cumulative_reward"] = round(self._episode["cumulative_reward"] + reward, 4)

        set_runtime_progress(self._state_dict())
        return self._build_observation(
            last_action_result=last_action_result,
            tool_output=tool_output,
            reward=round(reward, 4),
            done=self._episode["done"],
        )

    @property
    def state(self) -> UnifiedIncidentState:
        return UnifiedIncidentState(**self._state_dict())

    def _make_episode(self, scenario: dict[str, Any], episode_id: str | None = None) -> dict[str, Any]:
        services = {
            name: ServiceHealth(name=name, **payload)
            for name, payload in scenario["initial_services"].items()
        }
        checks = {
            "database_recovery": CheckResult(name="database_recovery", passed=False, detail="Database recovery has not been verified yet."),
            "end_to_end": CheckResult(name="end_to_end", passed=False, detail="End-to-end health has not been verified yet."),
        }
        return {
            "episode_id": episode_id or str(uuid.uuid4()),
            "scenario": scenario,
            "tick": 0,
            "step_count": 0,
            "max_ticks": scenario["max_ticks"],
            "difficulty": scenario["difficulty"],
            "services": services,
            "alerts": [Alert(**payload) for payload in scenario["initial_alerts"]],
            "discovered_evidence": [],
            "evidence_seen": set(),
            "recent_deploys": [scenario["deploy_history"]["worker"]],
            "checks": checks,
            "user_impact": 0.82,
            "slo_burn_rate": 0.91,
            "containment_applied": False,
            "cause_removed": False,
            "worker_isolated": False,
            "worker_version": "worker@2026.04.23-bad",
            "hypothesis_seen": set(),
            "failure_type": None,
            "why_failed": None,
            "loop_warning": None,
            "last_action_key": None,
            "repeat_count": 0,
            "incident_resolved": False,
            "workflow_stage": "triage",
            "cumulative_reward": 0.0,
            "wasteful_ticks": 0,
            "score_breakdown": {
                "recovery_score": 0.0,
                "containment_score": 0.0,
                "verification_score": 0.0,
                "impact_score": 0.0,
                "efficiency_score": 0.10,
                "final_score": 0.10,
            },
            "final_score": 0.10,
            "last_action_result": "",
            "done": False,
        }

    def _query_logs(self, service: str | None) -> str:
        assert service is not None
        return self._episode["scenario"]["logs"][service]

    def _query_metrics(self, service: str | None, metric: str | None) -> str:
        assert service is not None and metric is not None
        return self._episode["scenario"]["metrics"][service][metric]

    def _query_dependencies(self, service: str | None) -> str:
        assert service is not None
        return self._episode["scenario"]["dependencies"][service]

    def _query_deploys(self, service: str | None) -> str:
        assert service is not None
        return self._episode["scenario"]["deploy_history"][service]

    def _submit_hypothesis(self, action: UnifiedIncidentAction) -> tuple[float, bool, str]:
        assert action.hypothesis is not None
        normalized = json.dumps(action.hypothesis.model_dump(), sort_keys=True)
        if normalized in self._episode["hypothesis_seen"]:
            return 0.0, False, "Repeated hypothesis recorded with no additional reward."
        self._episode["hypothesis_seen"].add(normalized)
        truth = self._episode["scenario"]["truth"]
        payload = action.hypothesis
        cause_match = 1.0 if payload.root_cause == truth["root_cause"] else 0.0
        service_match = len(set(payload.affected_services) & set(truth["affected_services"])) / len(set(truth["affected_services"]))
        action_quality = 1.0 if payload.recommended_next_action == truth["best_next_action"] else -0.4
        if cause_match == 1.0:
            calibration = 1.0 if payload.confidence >= 0.7 else 0.5
        else:
            calibration = -1.0 if payload.confidence >= 0.7 else -0.2
        reward = (0.04 * cause_match) + (0.03 * service_match) + (0.03 * action_quality) + (0.02 * calibration)
        return round(reward, 4), True, "Hypothesis recorded. Reward reflects root-cause accuracy, service localization, confidence calibration, and next-action quality."

    def _rollback_deploy(self, service: str | None) -> tuple[bool, float, str]:
        assert service is not None
        if service != "worker":
            self._set_failure("wrong_remediation_target", "Rolling back a service without a causal link wastes time and risk.")
            return False, self._unsafe_penalty(), f"Rollback on {service} did not address the incident."
        if self._episode["cause_removed"]:
            return False, 0.0, "Worker deploy is already rolled back."
        self._episode["cause_removed"] = True
        self._episode["containment_applied"] = True
        self._episode["worker_version"] = "worker@2026.04.23-good"
        self._episode["services"]["worker"] = ServiceHealth(
            name="worker",
            status="healthy",
            cpu_pct=32.0,
            memory_pct=37.0,
            error_rate_pct=2.0,
            latency_ms=40.0,
        )
        self._episode["user_impact"] = min(self._episode["user_impact"], 0.55)
        self._episode["slo_burn_rate"] = min(self._episode["slo_burn_rate"], 0.58)
        return True, 0.0, "Rolled back the worker deploy; downstream load should now stabilize after dependent services recover."

    def _restart_service(self, service: str | None) -> tuple[bool, float, str]:
        assert service is not None
        if service == "database":
            if not self._episode["cause_removed"]:
                self._set_failure("premature_restart", "Restarting the database before removing the trigger only causes another crash loop.")
                return False, self._unsafe_penalty(), "Database restart failed because the worker is still driving overload."
            self._episode["services"]["database"] = ServiceHealth(
                name="database",
                status="healthy",
                cpu_pct=34.0,
                memory_pct=39.0,
                error_rate_pct=0.0,
                latency_ms=22.0,
            )
            self._episode["services"]["api-gateway"] = ServiceHealth(
                name="api-gateway",
                status="healthy",
                cpu_pct=28.0,
                memory_pct=31.0,
                error_rate_pct=0.0,
                latency_ms=38.0,
            )
            self._episode["user_impact"] = 0.14
            self._episode["slo_burn_rate"] = 0.18
            return True, 0.0, "Database restarted cleanly after the bad deploy was rolled back."
        self._set_failure("low_value_restart", f"Restarting {service} is not the safe next remediation step for this incident.")
        return False, self._unsafe_penalty() / 2, f"Restarting {service} had little or no positive effect."

    def _isolate_service(self, service: str | None) -> tuple[bool, float, str]:
        assert service is not None
        if service != "worker":
            self._set_failure("wrong_isolation_target", f"Isolating {service} does not contain the dominant failure path.")
            return False, self._unsafe_penalty() / 2, f"Isolation of {service} did not materially reduce blast radius."
        if self._episode["worker_isolated"]:
            return False, 0.0, "Worker is already isolated."
        self._episode["worker_isolated"] = True
        self._episode["containment_applied"] = True
        self._episode["services"]["worker"] = ServiceHealth(
            name="worker",
            status="isolated",
            cpu_pct=8.0,
            memory_pct=18.0,
            error_rate_pct=0.0,
            latency_ms=0.0,
        )
        self._episode["services"]["database"] = ServiceHealth(
            name="database",
            status="healthy",
            cpu_pct=41.0,
            memory_pct=46.0,
            error_rate_pct=0.0,
            latency_ms=26.0,
        )
        self._episode["services"]["api-gateway"] = ServiceHealth(
            name="api-gateway",
            status="degraded",
            cpu_pct=34.0,
            memory_pct=33.0,
            error_rate_pct=7.0,
            latency_ms=91.0,
        )
        self._episode["user_impact"] = 0.45
        self._episode["slo_burn_rate"] = 0.47
        return True, 0.0, "Worker isolated. Blast radius shrank, but end-to-end service remains degraded until the worker path is restored safely."

    def _run_check(self, check_name: str | None) -> tuple[str, bool, str]:
        assert check_name is not None
        if check_name == "database_recovery":
            passed = self._episode["services"]["database"].status == "healthy" and self._episode["cause_removed"]
            detail = (
                "Database is healthy and no longer crashing."
                if passed
                else "Database is still unstable or the triggering cause is still present."
            )
        else:
            passed = (
                self._episode["services"]["database"].status == "healthy"
                and self._episode["services"]["api-gateway"].status == "healthy"
                and self._episode["cause_removed"]
                and not self._episode["worker_isolated"]
            )
            detail = (
                "End-to-end login traffic is healthy."
                if passed
                else "End-to-end traffic still fails or remains degraded."
            )
        self._episode["checks"][check_name] = CheckResult(name=check_name, passed=passed, detail=detail)
        useful = self._mark_evidence_once(f"check:{check_name}:{passed}", detail)
        return detail, useful, f"Ran {check_name} check."

    def _declare_resolved(self) -> tuple[bool, float, float, str]:
        checks = self._episode["checks"]
        safe_to_resolve = checks["database_recovery"].passed and checks["end_to_end"].passed
        if not safe_to_resolve:
            self._set_failure("premature_resolution", "The incident is not verified as resolved yet.")
            return False, self._episode["scenario"]["reward_config"]["premature_resolution_penalty"], 0.0, "Resolution declaration rejected: required checks have not passed."
        self._episode["incident_resolved"] = True
        self._episode["done"] = True
        return True, 0.0, self._episode["scenario"]["reward_config"]["successful_resolution_bonus"], "Incident declared resolved after passing objective checks."

    def _mark_evidence_once(self, key: str, detail: str) -> bool:
        if key in self._episode["evidence_seen"]:
            return False
        self._episode["evidence_seen"].add(key)
        self._episode["discovered_evidence"].append(detail)
        return True

    def _unsafe_penalty(self) -> float:
        return float(self._episode["scenario"]["reward_config"]["unsafe_action_penalty"])

    def _set_failure(self, failure_type: str, why_failed: str) -> None:
        self._episode["failure_type"] = failure_type
        self._episode["why_failed"] = why_failed

    def _advance_world(self) -> None:
        if not self._episode["cause_removed"] and not self._episode["worker_isolated"]:
            self._episode["services"]["worker"] = ServiceHealth(
                name="worker",
                status="degraded",
                cpu_pct=88.0,
                memory_pct=71.0,
                error_rate_pct=19.0,
                latency_ms=420.0,
            )
            self._episode["services"]["database"] = ServiceHealth(
                name="database",
                status="crashed",
                cpu_pct=99.0,
                memory_pct=97.0,
                error_rate_pct=100.0,
                latency_ms=0.0,
            )
            self._episode["services"]["api-gateway"] = ServiceHealth(
                name="api-gateway",
                status="degraded",
                cpu_pct=61.0,
                memory_pct=38.0,
                error_rate_pct=24.0,
                latency_ms=640.0,
            )
            self._episode["user_impact"] = max(self._episode["user_impact"], 0.82)
            self._episode["slo_burn_rate"] = max(self._episode["slo_burn_rate"], 0.91)
        if self._episode["worker_isolated"] and not self._episode["cause_removed"]:
            self._episode["containment_applied"] = True
        self._episode["workflow_stage"] = self._workflow_stage()

    def _refresh_alerts(self) -> None:
        alerts: list[Alert] = []
        for service_name in SERVICE_ORDER:
            service = self._episode["services"][service_name]
            if service.status == "crashed":
                alerts.append(Alert(service=service_name, severity="critical", message=f"{service_name} is unavailable."))
            elif service.status == "degraded":
                alerts.append(Alert(service=service_name, severity="warning", message=f"{service_name} is degraded."))
        if self._episode["user_impact"] >= 0.3 and not any(alert.service == "api-gateway" for alert in alerts):
            alerts.append(Alert(service="api-gateway", severity="warning", message="User-visible impact remains elevated."))
        self._episode["alerts"] = alerts

    def _update_loop_feedback(self, action: UnifiedIncidentAction, progressed: bool) -> None:
        action_key = repr(action.model_dump(exclude_none=True))
        if progressed:
            self._episode["last_action_key"] = action_key
            self._episode["repeat_count"] = 0
            return
        if self._episode["last_action_key"] == action_key:
            self._episode["repeat_count"] += 1
        else:
            self._episode["repeat_count"] = 1
        self._episode["last_action_key"] = action_key
        if self._episode["repeat_count"] >= 2:
            self._episode["loop_warning"] = "The same no-progress action has repeated; choose a different evidence source or remediation step."

    def _workflow_stage(self) -> str:
        if self._episode["incident_resolved"]:
            return "resolved"
        checks = self._episode["checks"]
        if checks["database_recovery"].passed or checks["end_to_end"].passed:
            return "validation"
        if self._episode["containment_applied"] or self._episode["cause_removed"] or self._episode["worker_isolated"]:
            return "mitigation"
        return "triage"

    def _allowed_actions(self) -> list[str]:
        return list(ALL_ACTIONS)

    def _required_fields_by_action(self) -> dict[str, list[str]]:
        return {action: REQUIRED_FIELDS_BY_ACTION[action] for action in self._allowed_actions()}

    def _progress_flags(self) -> dict[str, bool]:
        checks = self._episode["checks"]
        return {
            "containment_applied": self._episode["containment_applied"],
            "cause_removed": self._episode["cause_removed"],
            "database_recovery": checks["database_recovery"].passed,
            "end_to_end": checks["end_to_end"].passed,
            "incident_resolved": self._episode["incident_resolved"],
        }

    def _incident_summary(self) -> str:
        return (
            "Gateway login traffic is failing because the worker is overloading the database after a recent worker deploy. "
            "Use evidence-gathering actions to diagnose, then choose a safe remediation and verify with explicit checks."
        )

    def _prompt_text(self, tool_output: str | None) -> str:
        lines = [
            f"TICK {self._episode['tick']}/{self._episode['max_ticks']}",
            f"WORKFLOW_STAGE: {self._episode['workflow_stage']}",
            "",
            "INCIDENT_SUMMARY:",
            self._incident_summary(),
            "",
            "ACTIVE_ALERTS:",
        ]
        if self._episode["alerts"]:
            lines.extend(f"- [{alert.severity.upper()}] {alert.service}: {alert.message}" for alert in self._episode["alerts"])
        else:
            lines.append("- none")
        lines.extend([
            "",
            "SERVICES:",
        ])
        for service_name in SERVICE_ORDER:
            health = self._episode["services"][service_name]
            lines.append(
                f"- {service_name}: {health.status} cpu={health.cpu_pct:.1f} mem={health.memory_pct:.1f} err={health.error_rate_pct:.1f} latency={health.latency_ms:.1f}"
            )
        lines.extend([
            "",
            f"USER_IMPACT: {self._episode['user_impact']:.2f}",
            f"SLO_BURN_RATE: {self._episode['slo_burn_rate']:.2f}",
            f"LAST_ACTION_RESULT: {self._episode['last_action_result'] or 'none'}",
            f"TOOL_OUTPUT: {tool_output or 'none'}",
            f"FAILURE_TYPE: {self._episode['failure_type'] or 'none'}",
            f"WHY_FAILED: {self._episode['why_failed'] or 'none'}",
            "",
            "CHECKS:",
        ])
        for check in self._episode["checks"].values():
            lines.append(f"- {check.name}: {'passed' if check.passed else 'pending'} - {check.detail}")
        lines.extend([
            "",
            "ALLOWED_ACTIONS:",
        ])
        lines.extend(f"- {action}" for action in self._allowed_actions())
        return "\n".join(lines)

    def _incident_health_potential(self) -> float:
        weights = self._episode["scenario"]["critical_service_weights"]
        services = self._episode["services"]
        operational = sum(weights.get(name, 0.0) * STATUS_VALUES[services[name].status] for name in weights)
        impact_relief = 1.0 - self._episode["user_impact"]
        burn_relief = 1.0 - self._episode["slo_burn_rate"]
        containment = 1.0 if self._episode["containment_applied"] else 0.0
        return round((0.55 * operational) + (0.2 * impact_relief) + (0.15 * burn_relief) + (0.10 * containment), 4)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self._episode["episode_id"],
            "step_count": self._episode["step_count"],
            "scenario_id": self._episode["scenario"]["id"],
            "difficulty": self._episode["difficulty"],
            "current_tick": self._episode["tick"],
            "max_ticks": self._episode["max_ticks"],
            "workflow_stage": self._episode["workflow_stage"],
            "active_alerts": [alert.model_dump() for alert in self._episode["alerts"]],
            "service_health": {name: service.model_dump() for name, service in self._episode["services"].items()},
            "discovered_evidence": list(self._episode["discovered_evidence"]),
            "recent_deploys": list(self._episode["recent_deploys"]),
            "checks": [check.model_dump() for check in self._episode["checks"].values()],
            "user_impact": self._episode["user_impact"],
            "slo_burn_rate": self._episode["slo_burn_rate"],
            "incident_resolved": self._episode["incident_resolved"],
            "containment_applied": self._episode["containment_applied"],
            "allowed_actions": self._allowed_actions(),
            "required_fields_by_action": self._required_fields_by_action(),
            "valid_action_example": None,
            "progress_flags": self._progress_flags(),
            "final_score": self._episode["final_score"],
            "score_breakdown": dict(self._episode["score_breakdown"]),
            "cumulative_reward": self._episode["cumulative_reward"],
            "wasteful_ticks": self._episode["wasteful_ticks"],
            "last_action_result": self._episode["last_action_result"],
            "failure_type": self._episode["failure_type"],
            "why_failed": self._episode["why_failed"],
        }

    def _build_observation(self, last_action_result: str, tool_output: str | None, reward: float, done: bool) -> UnifiedIncidentObservation:
        return UnifiedIncidentObservation(
            prompt_text=self._prompt_text(tool_output),
            incident_summary=self._incident_summary(),
            tick_count=self._episode["tick"],
            max_ticks=self._episode["max_ticks"],
            difficulty=self._episode["difficulty"],
            workflow_stage=self._episode["workflow_stage"],
            active_alerts=list(self._episode["alerts"]),
            service_health=dict(self._episode["services"]),
            discovered_evidence=list(self._episode["discovered_evidence"]),
            recent_deploys=list(self._episode["recent_deploys"]),
            checks=list(self._episode["checks"].values()),
            user_impact=self._episode["user_impact"],
            slo_burn_rate=self._episode["slo_burn_rate"],
            incident_resolved=self._episode["incident_resolved"],
            containment_applied=self._episode["containment_applied"],
            last_action_result=last_action_result,
            tool_output=tool_output,
            failure_type=self._episode["failure_type"],
            why_failed=self._episode["why_failed"],
            allowed_actions=self._allowed_actions(),
            required_fields_by_action=self._required_fields_by_action(),
            valid_action_example=None,
            common_trap=self._episode["scenario"].get("description"),
            loop_warning=self._episode["loop_warning"],
            blocked_until_security_complete=False,
            security_unlock_reason=None,
            best_recovery_action_family=None,
            progress_flags=self._progress_flags(),
            security_subquest_status=None,
            security_context={},
            final_score=self._episode["final_score"],
            score_breakdown=dict(self._episode["score_breakdown"]),
            reward=round(reward, 4),
            done=done,
        )
