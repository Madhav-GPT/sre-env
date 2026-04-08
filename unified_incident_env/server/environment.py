"""Core environment for the final deterministic preset pack."""

from __future__ import annotations

import json
import uuid
from typing import Any

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import (
    ActionType,
    Alert,
    SecurityContext,
    ServiceHealth,
    UnifiedIncidentAction,
    UnifiedIncidentObservation,
    UnifiedIncidentState,
)
from .challenge import DEFAULT_SCENARIO_ID, SCENARIOS, get_scenario, scenario_for_difficulty, set_runtime_progress
from .grader import UnifiedIncidentGrader

SERVICE_ORDER = ("api-gateway", "cache", "database", "worker")
STAGE_ALLOWED_ACTIONS: dict[str, list[ActionType]] = {
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
    "verification": ["submit_security_fix", "restart_service", "rollback_deploy"],
    "postmortem": ["submit_postmortem"],
    "done": [],
}
REQUIRED_FIELDS_BY_ACTION: dict[str, list[str]] = {
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


class UnifiedIncidentEnvironment(
    Environment[UnifiedIncidentAction, UnifiedIncidentObservation, UnifiedIncidentState]
):
    """Deterministic unified incident-response environment."""

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
                "A deterministic unified incident benchmark with one SRE flow, "
                "one security subquest, and one final score."
            ),
            version="1.0.0",
            author="Daksh Verma",
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> UnifiedIncidentObservation:
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

    def step(
        self,
        action: UnifiedIncidentAction | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> UnifiedIncidentObservation:
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
        self._clear_teaching_feedback()

        reward = 0.0
        tool_output: str | None = None
        progressed = False

        if action.action_type in {"query_logs", "query_metrics", "query_dependencies"}:
            reward, last_action_result, tool_output, progressed = self._handle_investigation(action)
        elif action.action_type in {"inspect_code", "classify_vulnerability", "apply_patch", "verify_security_fix", "submit_security_fix"}:
            reward, last_action_result, tool_output, progressed = self._handle_security(action)
        elif action.action_type in {"restart_service", "rollback_deploy"}:
            reward, last_action_result, progressed = self._handle_infrastructure(action)
        elif action.action_type == "submit_postmortem":
            reward, last_action_result, progressed = self._handle_postmortem(action)
        else:
            last_action_result = f"Unsupported action {action.action_type!r}."
            self._set_failure(
                failure_type="unsupported_action",
                why_failed="The action is not part of the public environment schema.",
            )

        if not progressed:
            self._episode["wasteful_ticks"] += 1
        self._update_loop_feedback(action, progressed)

        self._update_workflow_stage()
        self._episode["score_breakdown"] = self._grader.compute_breakdown(
            self._state_dict(),
            self._episode["scenario"],
        )
        self._episode["final_score"] = self._episode["score_breakdown"]["final_score"]
        self._episode["cumulative_reward"] = round(
            self._episode["cumulative_reward"] + reward,
            4,
        )
        self._episode["last_action_result"] = last_action_result

        done = False
        if self._episode["postmortem_submitted"]:
            self._episode["workflow_stage"] = "done"
            self._episode["done"] = True
            done = True
        elif self._episode["tick"] >= self._episode["max_ticks"]:
            self._episode["done"] = True
            done = True
            last_action_result = f"{last_action_result} Tick budget exhausted.".strip()
            self._episode["last_action_result"] = last_action_result

        set_runtime_progress(self._state_dict())
        return self._build_observation(
            last_action_result=last_action_result,
            tool_output=tool_output,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> UnifiedIncidentState:
        return UnifiedIncidentState(**self._state_dict())

    def _make_episode(self, scenario: dict, episode_id: str | None = None) -> dict[str, Any]:
        services = {
            name: ServiceHealth(name=name, **payload)
            for name, payload in scenario["initial_services"].items()
        }
        alerts = [Alert(**payload) for payload in scenario["initial_alerts"]]
        return {
            "episode_id": episode_id or str(uuid.uuid4()),
            "scenario": scenario,
            "tick": 0,
            "step_count": 0,
            "max_ticks": scenario["max_ticks"],
            "difficulty": scenario["difficulty"],
            "services": services,
            "alerts": alerts,
            "discovered_evidence": [],
            "matched_evidence_ids": set(),
            "relevant_investigations": 0,
            "identified_root_cause": None,
            "failure_type": None,
            "why_failed": None,
            "common_trap": scenario.get("common_trap"),
            "loop_warning": None,
            "blocked_until_security_complete": False,
            "security_unlock_reason": None,
            "best_recovery_action_family": None,
            "last_action_key": None,
            "repeated_no_progress_count": 0,
            "security_subquest_status": "locked",
            "security_context": SecurityContext(
                code_visible=False,
                selected_vulnerability=None,
                selected_patch=None,
                exploit_blocked=None,
                functionality_preserved=None,
            ),
            "security_fix_submitted": False,
            "incident_resolved": False,
            "postmortem_submitted": False,
            "postmortem_score": 0.0,
            "infra_progress": 0,
            "correct_infra_steps": 0,
            "infra_restored_in_correct_order": False,
            "workflow_stage": "diagnosis",
            "wasteful_ticks": 0,
            "cumulative_reward": 0.0,
            "final_score": 0.10,
            "score_breakdown": {
                "infrastructure_score": 0.0,
                "security_score": 0.0,
                "efficiency_score": 0.10,
                "postmortem_score": 0.0,
                "final_score": 0.10,
            },
            "last_action_result": "",
            "done": False,
        }

    def _clear_teaching_feedback(self) -> None:
        self._episode["failure_type"] = None
        self._episode["why_failed"] = None
        self._episode["loop_warning"] = None
        self._episode["blocked_until_security_complete"] = False
        self._episode["best_recovery_action_family"] = None

    def _set_failure(
        self,
        *,
        failure_type: str,
        why_failed: str,
        blocked_until_security_complete: bool = False,
        best_recovery_action_family: str | None = None,
    ) -> None:
        self._episode["failure_type"] = failure_type
        self._episode["why_failed"] = why_failed
        self._episode["blocked_until_security_complete"] = (
            blocked_until_security_complete
        )
        self._episode["best_recovery_action_family"] = best_recovery_action_family

    def _handle_investigation(
        self, action: UnifiedIncidentAction
    ) -> tuple[float, str, str | None, bool]:
        scenario = self._episode["scenario"]
        service = action.service
        assert service is not None

        if action.action_type == "query_logs":
            output = scenario["logs"].get(service, f"No logs available for {service}.")
        elif action.action_type == "query_metrics":
            output = scenario["metrics"].get(service, {}).get(
                action.metric or "",
                f"No metric {action.metric!r} available for {service}.",
            )
        else:
            output = scenario["dependencies"].get(
                service,
                f"No dependency data available for {service}.",
            )

        reward = 0.0
        progressed = False
        for rule in scenario["evidence_rules"]:
            if rule["action_type"] != action.action_type:
                continue
            if rule["service"] != service:
                continue
            if action.action_type == "query_metrics" and rule.get("metric") != action.metric:
                continue
            if rule["id"] in self._episode["matched_evidence_ids"]:
                break
            self._episode["matched_evidence_ids"].add(rule["id"])
            self._episode["discovered_evidence"].append(rule["detail"])
            self._episode["relevant_investigations"] += 1
            reward += 0.05
            progressed = True
            if rule.get("identifies_root_cause"):
                self._episode["identified_root_cause"] = scenario["root_cause"]
            if rule.get("unlocks_security"):
                self._episode["security_subquest_status"] = "active"
                self._episode["security_unlock_reason"] = rule.get(
                    "unlock_reason",
                    "Relevant incident evidence unlocked the security subquest.",
                )
            break

        if (
            self._episode["security_subquest_status"] == "locked"
            and len(self._episode["matched_evidence_ids"]) >= scenario["unlock_threshold"]
        ):
            self._episode["security_subquest_status"] = "active"
            self._episode["security_unlock_reason"] = (
                "Enough incident evidence was collected to justify security investigation."
            )

        return (
            round(reward, 4),
            f"{action.action_type} returned data for {service}.",
            output,
            progressed,
        )

    def _handle_security(
        self, action: UnifiedIncidentAction
    ) -> tuple[float, str, str | None, bool]:
        security = self._episode["security_context"]
        scenario_security = self._episode["scenario"]["security"]
        status = self._episode["security_subquest_status"]

        if status == "locked":
            self._set_failure(
                failure_type="security_locked",
                why_failed="Security actions are ineffective until the subquest is unlocked by relevant incident evidence.",
                best_recovery_action_family="query_logs",
            )
            return (
                0.0,
                "Security subquest is locked. Investigate the incident first.",
                None,
                False,
            )

        if action.action_type == "inspect_code":
            if security.code_visible:
                self._set_failure(
                    failure_type="code_already_visible",
                    why_failed="Inspecting code again does not reveal new information.",
                    best_recovery_action_family="classify_vulnerability",
                )
                return 0.0, "Code is already visible.", None, False
            security.code_visible = True
            return (
                0.0,
                "Inspected the vulnerable code path.",
                (
                    f"{scenario_security['code_context']}\n"
                    f"Patch options: {', '.join(option['id'] for option in scenario_security['patch_options'])}"
                ),
                True,
            )

        if action.action_type == "classify_vulnerability":
            if not security.code_visible:
                self._set_failure(
                    failure_type="inspect_required",
                    why_failed="Classification is unreliable until the vulnerable code is visible.",
                    best_recovery_action_family="inspect_code",
                )
                return 0.0, "Inspect the code before classifying.", None, False
            security.selected_vulnerability = action.vulnerability_type
            if action.vulnerability_type == scenario_security["correct_vulnerability"]:
                return 0.10, "Correct vulnerability classification.", None, True
            self._set_failure(
                failure_type="wrong_vulnerability",
                why_failed="The selected vulnerability does not match the exploit described by the current evidence.",
                best_recovery_action_family="classify_vulnerability",
            )
            return -0.08, "Wrong vulnerability classification.", None, False

        if action.action_type == "apply_patch":
            if not security.code_visible:
                self._set_failure(
                    failure_type="inspect_required",
                    why_failed="Patching before inspecting code is unlikely to address the root cause safely.",
                    best_recovery_action_family="inspect_code",
                )
                return 0.0, "Inspect the code before patching.", None, False
            security.selected_patch = action.patch_id
            security.exploit_blocked = None
            security.functionality_preserved = None
            if action.patch_id == scenario_security["correct_patch"]:
                return 0.10, "Applied the correct patch.", None, True
            self._set_failure(
                failure_type="wrong_patch",
                why_failed="The selected patch does not safely close the active exploit path.",
                best_recovery_action_family="apply_patch",
            )
            return -0.10, f"Wrong patch applied: {action.patch_id}.", None, False

        if action.action_type == "verify_security_fix":
            if security.selected_patch is None:
                self._set_failure(
                    failure_type="verify_too_early",
                    why_failed="Verification needs a selected patch first.",
                    best_recovery_action_family="apply_patch",
                )
                return -0.05, "Verify called too early; no patch has been applied.", None, False
            outcome = scenario_security["verify_outcomes"].get(
                security.selected_patch,
                {
                    "exploit_blocked": False,
                    "functionality_preserved": False,
                    "message": f"Verification failed: {security.selected_patch} is an invalid patch for this scenario.",
                },
            )
            security.exploit_blocked = outcome["exploit_blocked"]
            security.functionality_preserved = outcome["functionality_preserved"]
            if outcome["exploit_blocked"] and outcome["functionality_preserved"]:
                return 0.10, outcome["message"], outcome["message"], True
            self._set_failure(
                failure_type="verification_failed",
                why_failed=outcome["message"],
                best_recovery_action_family="apply_patch",
            )
            return 0.0, outcome["message"], outcome["message"], False

        if action.action_type == "submit_security_fix":
            if not (
                security.selected_vulnerability == scenario_security["correct_vulnerability"]
                and security.selected_patch == scenario_security["correct_patch"]
                and security.exploit_blocked is True
                and security.functionality_preserved is True
            ):
                self._set_failure(
                    failure_type="submit_too_early",
                    why_failed="Security submission only works after correct classification, correct patching, and successful verification.",
                    best_recovery_action_family="verify_security_fix",
                )
                return -0.05, "Submit security fix called before a successful verify.", None, False
            self._episode["security_fix_submitted"] = True
            self._episode["security_subquest_status"] = "completed"
            return 0.05, "Security subquest completed.", None, True

        return 0.0, "Unsupported security action.", None, False

    def _handle_infrastructure(
        self, action: UnifiedIncidentAction
    ) -> tuple[float, str, bool]:
        scenario = self._episode["scenario"]
        service = action.service
        assert service is not None

        if self._episode["infra_progress"] >= len(scenario["recovery_sequence"]):
            self._set_failure(
                failure_type="infra_already_complete",
                why_failed="No further infrastructure recovery actions are required right now.",
                best_recovery_action_family="submit_postmortem",
            )
            return -0.08, "No infrastructure recovery steps remain.", False

        expected = scenario["recovery_sequence"][self._episode["infra_progress"]]
        matches_expected = (
            action.action_type == expected["action_type"] and service == expected["service"]
        )
        if matches_expected and expected.get("requires_security_completion") and self._episode["security_subquest_status"] != "completed":
            trap_message = self._trap_message(scenario, action)
            if trap_message is not None:
                self._set_failure(
                    failure_type="infra_before_security",
                    why_failed=trap_message,
                    blocked_until_security_complete=True,
                    best_recovery_action_family="submit_security_fix",
                )
                return -0.10, trap_message, False
            self._set_failure(
                failure_type="infra_before_security",
                why_failed="This infrastructure step is blocked until the security subquest is completed.",
                blocked_until_security_complete=True,
                best_recovery_action_family="submit_security_fix",
            )
            return -0.08, "Infrastructure step attempted before security completion.", False

        if not matches_expected:
            trap_message = self._trap_message(scenario, action)
            if trap_message is not None:
                self._set_failure(
                    failure_type="trap_action",
                    why_failed=trap_message,
                    best_recovery_action_family=expected["action_type"],
                )
                return -0.10, trap_message, False
            self._set_failure(
                failure_type="wrong_infra_action",
                why_failed="This restart or rollback does not address the current bottleneck.",
                best_recovery_action_family=expected["action_type"],
            )
            return -0.08, "Wrong restart or rollback for the current recovery step.", False

        for service_name, payload in expected["updates"].items():
            self._episode["services"][service_name] = ServiceHealth(
                name=service_name,
                **payload,
            )
        self._refresh_alerts()
        self._episode["infra_progress"] += 1
        self._episode["correct_infra_steps"] += 1

        reward = 0.10
        if self._episode["infra_progress"] == len(scenario["recovery_sequence"]):
            self._episode["infra_restored_in_correct_order"] = True
            if self._all_required_services_healthy():
                self._episode["incident_resolved"] = True
                reward += 0.30
        return reward, expected["message"], True

    def _trap_message(
        self,
        scenario: dict[str, Any],
        action: UnifiedIncidentAction,
    ) -> str | None:
        for trap in scenario["trap_actions"]:
            if trap["action_type"] != action.action_type or trap["service"] != action.service:
                continue
            if trap.get("requires_security_incomplete") and self._episode["security_subquest_status"] == "completed":
                continue
            return trap["message"]
        return None

    def _handle_postmortem(
        self, action: UnifiedIncidentAction
    ) -> tuple[float, str, bool]:
        if not self._episode["incident_resolved"]:
            self._set_failure(
                failure_type="postmortem_too_early",
                why_failed="Postmortem submission is only valid after services are healthy and recovery is complete.",
                best_recovery_action_family="restart_service",
            )
            return -0.10, "Submit postmortem only after full recovery.", False
        if self._episode["postmortem_submitted"]:
            self._set_failure(
                failure_type="postmortem_already_submitted",
                why_failed="The postmortem is already complete for this episode.",
            )
            return 0.0, "Postmortem already submitted.", False
        assert action.postmortem is not None
        postmortem_score = self._grader.postmortem_score(
            action.postmortem,
            self._episode["scenario"],
        )
        self._episode["postmortem_submitted"] = True
        self._episode["postmortem_score"] = postmortem_score
        return postmortem_score, "Postmortem submitted.", True

    def _all_required_services_healthy(self) -> bool:
        return all(
            self._episode["services"][name].status == "healthy"
            for name in SERVICE_ORDER
        )

    def _refresh_alerts(self) -> None:
        self._episode["alerts"] = [
            alert
            for alert in self._episode["alerts"]
            if self._episode["services"][alert.service].status != "healthy"
        ]

    def _update_loop_feedback(
        self,
        action: UnifiedIncidentAction,
        progressed: bool,
    ) -> None:
        action_key = repr(action.model_dump(exclude_none=True))
        if progressed:
            self._episode["last_action_key"] = action_key
            self._episode["repeated_no_progress_count"] = 0
            return

        if self._episode["last_action_key"] == action_key:
            self._episode["repeated_no_progress_count"] += 1
        else:
            self._episode["repeated_no_progress_count"] = 1
        self._episode["last_action_key"] = action_key

        if self._episode["repeated_no_progress_count"] >= 2:
            repeat_count = self._episode["repeated_no_progress_count"]
            self._episode["loop_warning"] = (
                f"The same no-progress action has repeated {repeat_count} times."
                " Stop repeating it; choose a different allowed action or move to the next workflow stage."
            )
            if self._episode["failure_type"] is None:
                self._set_failure(
                    failure_type="repeated_no_progress_action",
                    why_failed="The system is not progressing because the same ineffective action keeps repeating.",
                )
            if repeat_count >= 3:
                self._episode["cumulative_reward"] = round(
                    self._episode["cumulative_reward"] - 0.02,
                    4,
                )

    def _update_workflow_stage(self) -> None:
        if self._episode["postmortem_submitted"]:
            self._episode["workflow_stage"] = "done"
            return
        if self._episode["incident_resolved"]:
            self._episode["workflow_stage"] = "postmortem"
            return
        if self._episode["security_subquest_status"] == "completed":
            self._episode["workflow_stage"] = "remediation"
            return
        security = self._episode["security_context"]
        if security.exploit_blocked is True and security.functionality_preserved is True:
            self._episode["workflow_stage"] = "verification"
            return
        if self._episode["security_subquest_status"] == "active":
            self._episode["workflow_stage"] = "security_subquest"
            return
        if self._episode["identified_root_cause"]:
            self._episode["workflow_stage"] = "root_cause_analysis"
            return
        self._episode["workflow_stage"] = "diagnosis"

    def _allowed_actions(self) -> list[str]:
        return list(STAGE_ALLOWED_ACTIONS.get(self._episode["workflow_stage"], []))

    def _required_fields_by_action(self) -> dict[str, list[str]]:
        return {
            action: REQUIRED_FIELDS_BY_ACTION[action]
            for action in self._allowed_actions()
        }

    def _valid_action_example(self) -> dict[str, Any]:
        stage = self._episode["workflow_stage"]
        security = self._episode["security_context"]
        scenario = self._episode["scenario"]
        if stage in {"diagnosis", "root_cause_analysis"}:
            if scenario["id"] == "database_sqli_outage":
                return {"action_type": "query_logs", "service": "database"}
            if scenario["id"] == "cache_abuse_broken_access_control":
                return {
                    "action_type": "query_metrics",
                    "service": "cache",
                    "metric": "cpu",
                }
            return {"action_type": "query_logs", "service": "worker"}
        if stage == "security_subquest":
            if not security.code_visible:
                return {"action_type": "inspect_code"}
            if security.selected_vulnerability is None:
                return {
                    "action_type": "classify_vulnerability",
                    "vulnerability_type": scenario["security"]["correct_vulnerability"],
                }
            if security.selected_patch is None:
                return {
                    "action_type": "apply_patch",
                    "patch_id": scenario["security"]["correct_patch"],
                }
            if security.exploit_blocked is not True:
                return {"action_type": "verify_security_fix"}
            return {"action_type": "submit_security_fix"}
        if stage in {"remediation", "verification"}:
            if (
                stage == "verification"
                and not self._episode["security_fix_submitted"]
                and security.exploit_blocked is True
                and security.functionality_preserved is True
            ):
                return {"action_type": "submit_security_fix"}
            next_index = min(
                self._episode["infra_progress"],
                len(scenario["recovery_sequence"]) - 1,
            )
            expected = scenario["recovery_sequence"][next_index]
            return {
                "action_type": expected["action_type"],
                "service": expected["service"],
            }
        return {
            "action_type": "submit_postmortem",
            "postmortem": {
                "root_cause": scenario["root_cause"],
                "attack_vector": scenario["attack_vector"],
                "timeline": ["Investigated", "Patched", "Recovered"],
                "remediation_steps": ["Patch", "Recover"],
                "prevention_steps": ["Detect", "Harden"],
            },
        }

    def _progress_flags(self) -> dict[str, bool]:
        return {
            "root_cause_identified": self._episode["identified_root_cause"] is not None,
            "security_subquest_unlocked": self._episode["security_subquest_status"] != "locked",
            "code_visible": self._episode["security_context"].code_visible,
            "security_fix_submitted": self._episode["security_fix_submitted"],
            "incident_resolved": self._episode["incident_resolved"],
            "postmortem_submitted": self._episode["postmortem_submitted"],
        }

    def _stage_hint(self) -> str:
        stage = self._episode["workflow_stage"]
        if stage == "diagnosis":
            return "Use investigation actions to identify the root cause quickly."
        if stage == "root_cause_analysis":
            return "Confirm the root cause and avoid repeated broad investigation."
        if stage == "security_subquest":
            return "Complete the security subquest with the next security action, then recover the system."
        if stage == "remediation":
            return "Recover the system health using the allowed remediation action."
        if stage == "verification":
            return "Verify the security fix and system recovery before submitting the security fix."
        if stage == "postmortem":
            return "Submit the postmortem now that the incident is resolved."
        return "Follow the current stage goal and allowed actions."

    def _stop_investigating_reason(self) -> str | None:
        if self._episode["loop_warning"]:
            return "Repeated no-progress actions detected. Stop investigating in the same way."
        if self._episode["workflow_stage"] == "root_cause_analysis":
            return "Confirm the root cause without broad additional queries."
        if self._episode["workflow_stage"] in {"security_subquest", "remediation", "verification", "postmortem"}:
            return "Avoid more query_* investigation actions unless strictly required by the current stage."
        return None

    def _patch_ids(self) -> list[str]:
        scenario_security = self._episode["scenario"].get("security") or {}
        patch_options = scenario_security.get("patch_options") or []
        return [option["id"] for option in patch_options if "id" in option]

    def _prompt_text(self, tool_output: str | None) -> str:
        allowed_actions = self._allowed_actions()
        required_fields = self._required_fields_by_action()
        valid_action_example = self._valid_action_example()
        progress_flags = self._progress_flags()
        lines = [
            f"TICK {self._episode['tick']}/{self._episode['max_ticks']}",
            f"WORKFLOW_STAGE: {self._episode['workflow_stage']}",
            "",
            "ACTIVE_ALERTS:",
        ]
        if self._episode["alerts"]:
            for alert in self._episode["alerts"]:
                lines.append(f"- [{alert.severity.upper()}] {alert.service}: {alert.message}")
        else:
            lines.append("- none")
        lines.extend(["", "SERVICES:"])
        for service_name in SERVICE_ORDER:
            health = self._episode["services"][service_name]
            lines.append(
                f"- {service_name}: {health.status} cpu={health.cpu_pct:.1f} mem={health.memory_pct:.1f} err={health.error_rate_pct:.1f}"
            )
        lines.extend(
            [
                "",
                "LAST_ACTION_RESULT:",
                self._episode["last_action_result"] or "none",
                "",
                "TOOL_OUTPUT:",
                tool_output or "none",
                "",
                "FAILURE_TYPE:",
                self._episode["failure_type"] or "none",
                "",
                "WHY_FAILED:",
                self._episode["why_failed"] or "none",
                "",
                "SECURITY_SUBQUEST_STATUS:",
                f"- {self._episode['security_subquest_status']}",
                "",
                "SECURITY_CONTEXT:",
            ]
        )
        security = self._episode["security_context"]
        lines.extend(
            [
                f"- code_visible: {'true' if security.code_visible else 'false'}",
                f"- selected_vulnerability: {security.selected_vulnerability or 'null'}",
                f"- selected_patch: {security.selected_patch or 'null'}",
                f"- exploit_blocked: {self._bool_or_null(security.exploit_blocked)}",
                f"- functionality_preserved: {self._bool_or_null(security.functionality_preserved)}",
                "",
                "ALLOWED_ACTIONS:",
            ]
        )
        lines.extend([f"- {action}" for action in allowed_actions])
        lines.extend(
            [
                "",
                "REQUIRED_FIELDS_BY_ACTION:",
            ]
        )
        lines.extend(
            [
                f"- {action}: {', '.join(fields) if fields else '(no extra fields)'}"
                for action, fields in required_fields.items()
            ]
        )
        stage_hint = self._stage_hint()
        stop_investigation = self._stop_investigating_reason()
        patch_ids = self._patch_ids()
        lines.extend(
            [
                "",
                "VALID_ACTION_EXAMPLE:",
                json.dumps(valid_action_example, separators=(",", ":")),
                "",
                "COMMON_TRAP:",
                self._episode["common_trap"] or "none",
                "",
                "STAGE_HINT:",
                stage_hint,
                "",
                "STOP_INVESTIGATING:",
                stop_investigation or "none",
            ]
        )
        if patch_ids and "apply_patch" in allowed_actions:
            lines.extend([
                "",
                "PATCH_IDS:",
                ", ".join(patch_ids),
            ])
        lines.extend(
            [
                "",
                "LOOP_WARNING:",
                self._episode["loop_warning"] or "none",
                "",
                "SECURITY_UNLOCK_REASON:",
                self._episode["security_unlock_reason"] or "none",
                "",
                "PROGRESS_FLAGS:",
            ]
        )
        lines.extend(
            [
                f"- {name}: {'true' if value else 'false'}"
                for name, value in progress_flags.items()
            ]
        )
        lines.extend(
            [
                "",
                "What is the next action? JSON only.",
            ]
        )
        return "\n".join(lines)

    def _bool_or_null(self, value: bool | None) -> str:
        if value is None:
            return "null"
        return "true" if value else "false"

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
            "service_health": {
                name: service.model_dump()
                for name, service in self._episode["services"].items()
            },
            "discovered_evidence": list(self._episode["discovered_evidence"]),
            "identified_root_cause": self._episode["identified_root_cause"],
            "failure_type": self._episode["failure_type"],
            "why_failed": self._episode["why_failed"],
            "allowed_actions": self._allowed_actions(),
            "required_fields_by_action": self._required_fields_by_action(),
            "valid_action_example": self._valid_action_example(),
            "common_trap": self._episode["common_trap"],
            "loop_warning": self._episode["loop_warning"],
            "blocked_until_security_complete": self._episode["blocked_until_security_complete"],
            "security_unlock_reason": self._episode["security_unlock_reason"],
            "best_recovery_action_family": self._episode["best_recovery_action_family"],
            "progress_flags": self._progress_flags(),
            "security_subquest_status": self._episode["security_subquest_status"],
            "security_context": self._episode["security_context"].model_dump(),
            "security_fix_submitted": self._episode["security_fix_submitted"],
            "incident_resolved": self._episode["incident_resolved"],
            "postmortem_submitted": self._episode["postmortem_submitted"],
            "cumulative_reward": self._episode["cumulative_reward"],
            "cumulative_score": self._episode["final_score"],
            "score_breakdown": {
                "infrastructure_score": self._episode["score_breakdown"]["infrastructure_score"],
                "security_score": self._episode["score_breakdown"]["security_score"],
                "efficiency_score": self._episode["score_breakdown"]["efficiency_score"],
                "postmortem_score": self._episode["postmortem_score"],
            },
            "relevant_investigations": self._episode["relevant_investigations"],
            "correct_infra_steps": self._episode["correct_infra_steps"],
            "infra_restored_in_correct_order": self._episode["infra_restored_in_correct_order"],
            "selected_vulnerability": self._episode["security_context"].selected_vulnerability,
            "selected_patch": self._episode["security_context"].selected_patch,
            "exploit_blocked": self._episode["security_context"].exploit_blocked,
            "functionality_preserved": self._episode["security_context"].functionality_preserved,
            "wasteful_ticks": self._episode["wasteful_ticks"],
            "last_action_result": self._episode["last_action_result"],
        }

    def _build_observation(
        self,
        last_action_result: str,
        tool_output: str | None,
        reward: float,
        done: bool,
    ) -> UnifiedIncidentObservation:
        return UnifiedIncidentObservation(
            prompt_text=self._prompt_text(tool_output),
            tick_count=self._episode["tick"],
            max_ticks=self._episode["max_ticks"],
            difficulty=self._episode["difficulty"],
            workflow_stage=self._episode["workflow_stage"],
            active_alerts=list(self._episode["alerts"]),
            service_health=dict(self._episode["services"]),
            last_action_result=last_action_result,
            tool_output=tool_output,
            failure_type=self._episode["failure_type"],
            why_failed=self._episode["why_failed"],
            allowed_actions=self._allowed_actions(),
            required_fields_by_action=self._required_fields_by_action(),
            valid_action_example=self._valid_action_example(),
            common_trap=self._episode["common_trap"],
            loop_warning=self._episode["loop_warning"],
            blocked_until_security_complete=self._episode["blocked_until_security_complete"],
            security_unlock_reason=self._episode["security_unlock_reason"],
            best_recovery_action_family=self._episode["best_recovery_action_family"],
            progress_flags=self._progress_flags(),
            security_subquest_status=self._episode["security_subquest_status"],
            security_context=self._episode["security_context"].model_copy(deep=True),
            final_score=self._episode["final_score"],
            score_breakdown={
                "infrastructure_score": self._episode["score_breakdown"]["infrastructure_score"],
                "security_score": self._episode["score_breakdown"]["security_score"],
                "efficiency_score": self._episode["score_breakdown"]["efficiency_score"],
                "postmortem_score": self._episode["postmortem_score"],
            },
            incident_resolved=self._episode["incident_resolved"],
            reward=round(reward, 4),
            done=done,
        )
