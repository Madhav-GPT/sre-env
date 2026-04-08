"""Typed models for the final preset-based unified incident environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

ActionType = Literal[
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
Difficulty = Literal["easy", "medium", "hard"]
MetricName = Literal["cpu", "memory", "latency", "error_rate", "throughput"]
ServiceName = Literal["api-gateway", "cache", "database", "worker"]
ServiceStatus = Literal["healthy", "degraded", "crashed"]
WorkflowStage = Literal[
    "diagnosis",
    "root_cause_analysis",
    "security_subquest",
    "remediation",
    "verification",
    "postmortem",
    "done",
]
SecuritySubquestStatus = Literal["locked", "active", "completed"]
VulnerabilityType = Literal[
    "sql_injection",
    "broken_access_control",
    "command_injection",
]


class PostmortemPayload(BaseModel):
    """Structured postmortem payload used by the final action."""

    model_config = ConfigDict(extra="forbid")

    root_cause: str = ""
    attack_vector: str = ""
    timeline: list[str] = Field(default_factory=list)
    remediation_steps: list[str] = Field(default_factory=list)
    prevention_steps: list[str] = Field(default_factory=list)


class UnifiedIncidentAction(Action):
    """One structured environment action."""

    model_config = ConfigDict(extra="ignore")

    action_type: ActionType
    service: ServiceName | None = None
    metric: MetricName | None = None
    vulnerability_type: VulnerabilityType | None = None
    patch_id: str | None = None
    postmortem: PostmortemPayload | None = None

    @model_validator(mode="before")
    @classmethod
    def _autofill_common_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        action_type = data.get("action_type")
        if action_type == "query_logs" and not data.get("service"):
            filled = dict(data)
            filled["service"] = "database"
            return filled
        return data

    @model_validator(mode="after")
    def _validate_payload(self) -> "UnifiedIncidentAction":
        if self.action_type in {
            "query_logs",
            "query_dependencies",
            "restart_service",
            "rollback_deploy",
        } and not self.service:
            raise PydanticCustomError(
                "missing_service",
                "service is required for {action_type}",
                {"action_type": self.action_type},
            )
        if self.action_type == "query_metrics":
            if not self.service:
                raise PydanticCustomError(
                    "missing_service",
                    "service is required for {action_type}",
                    {"action_type": "query_metrics"},
                )
            if not self.metric:
                raise PydanticCustomError(
                    "missing_metric",
                    "metric is required for {action_type}",
                    {"action_type": "query_metrics"},
                )
        if (
            self.action_type == "classify_vulnerability"
            and self.vulnerability_type is None
        ):
            raise PydanticCustomError(
                "missing_vulnerability_type",
                "vulnerability_type is required for {action_type}",
                {"action_type": "classify_vulnerability"},
            )
        if self.action_type == "apply_patch" and not self.patch_id:
            raise PydanticCustomError(
                "missing_patch_id",
                "patch_id is required for {action_type}",
                {"action_type": "apply_patch"},
            )
        if self.action_type == "submit_postmortem" and self.postmortem is None:
            raise PydanticCustomError(
                "missing_postmortem",
                "postmortem is required for {action_type}",
                {"action_type": "submit_postmortem"},
            )
        return self


class ServiceHealth(BaseModel):
    """Health snapshot for a service."""

    model_config = ConfigDict(extra="forbid")

    name: ServiceName
    status: ServiceStatus
    cpu_pct: float = Field(ge=0.0, le=100.0)
    memory_pct: float = Field(ge=0.0, le=100.0)
    error_rate_pct: float = Field(ge=0.0, le=100.0)
    latency_ms: float = Field(ge=0.0)


class Alert(BaseModel):
    """Alert exposed to the agent."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceName
    severity: Literal["warning", "critical"]
    message: str


class SecurityContext(BaseModel):
    """Structured security subquest context."""

    model_config = ConfigDict(extra="forbid")

    code_visible: bool = False
    selected_vulnerability: VulnerabilityType | None = None
    selected_patch: str | None = None
    exploit_blocked: bool | None = None
    functionality_preserved: bool | None = None


class PatchOption(BaseModel):
    """One security patch option."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str


class UnifiedIncidentObservation(Observation):
    """Observation returned after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    tick_count: int
    max_ticks: int
    difficulty: Difficulty
    workflow_stage: WorkflowStage
    active_alerts: list[Alert] = Field(default_factory=list)
    service_health: dict[str, ServiceHealth] = Field(default_factory=dict)
    last_action_result: str = ""
    tool_output: str | None = None
    failure_type: str | None = None
    why_failed: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    required_fields_by_action: dict[str, list[str]] = Field(default_factory=dict)
    valid_action_example: dict[str, Any] | None = None
    common_trap: str | None = None
    loop_warning: str | None = None
    blocked_until_security_complete: bool = False
    security_unlock_reason: str | None = None
    best_recovery_action_family: str | None = None
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    security_subquest_status: SecuritySubquestStatus = "locked"
    security_context: SecurityContext = Field(default_factory=SecurityContext)
    final_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    incident_resolved: bool = False
    reward: float = 0.0
    done: bool = False


class UnifiedIncidentState(State):
    """Persistent episode state."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step_count: int
    scenario_id: str
    difficulty: Difficulty
    current_tick: int
    max_ticks: int
    workflow_stage: WorkflowStage
    active_alerts: list[Alert] = Field(default_factory=list)
    service_health: dict[str, ServiceHealth] = Field(default_factory=dict)
    discovered_evidence: list[str] = Field(default_factory=list)
    identified_root_cause: str | None = None
    failure_type: str | None = None
    why_failed: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    required_fields_by_action: dict[str, list[str]] = Field(default_factory=dict)
    valid_action_example: dict[str, Any] | None = None
    common_trap: str | None = None
    loop_warning: str | None = None
    blocked_until_security_complete: bool = False
    security_unlock_reason: str | None = None
    best_recovery_action_family: str | None = None
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    security_subquest_status: SecuritySubquestStatus = "locked"
    security_context: SecurityContext = Field(default_factory=SecurityContext)
    relevant_investigations: int = 0
    correct_infra_steps: int = 0
    infra_restored_in_correct_order: bool = False
    selected_vulnerability: VulnerabilityType | None = None
    selected_patch: str | None = None
    exploit_blocked: bool | None = None
    functionality_preserved: bool | None = None
    security_fix_submitted: bool = False
    incident_resolved: bool = False
    postmortem_submitted: bool = False
    cumulative_reward: float = 0.0
    cumulative_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    wasteful_ticks: int = 0
    last_action_result: str = ""


class ScenarioSummary(BaseModel):
    """Public scenario summary."""

    model_config = ConfigDict(extra="forbid")

    id: str
    difficulty: Difficulty
    name: str
    description: str
    root_cause: str
    optimal_ticks: int


class ScenarioCatalog(BaseModel):
    """Public scenario catalog."""

    model_config = ConfigDict(extra="forbid")

    environment: str = "unified_incident_env"
    default_scenario_id: str
    available_difficulties: list[Difficulty]
    filtered_difficulty: Difficulty | None = None
    scenarios: list[ScenarioSummary]


class BaselineStep(BaseModel):
    """One baseline action."""

    model_config = ConfigDict(extra="forbid")

    action: UnifiedIncidentAction
    rationale: str = ""


class BaselineDefinition(BaseModel):
    """One baseline trajectory."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    description: str
    optimal_ticks: int
    actions: list[BaselineStep] = Field(default_factory=list)


class BaselineCatalog(BaseModel):
    """Public baseline catalog."""

    model_config = ConfigDict(extra="forbid")

    environment: str = "unified_incident_env"
    baselines: list[BaselineDefinition]


class GraderCheck(BaseModel):
    """One normalized grader check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str
    weight: float


class GraderReport(BaseModel):
    """Episode-grade report."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    message: str
    breakdown: dict[str, float] = Field(default_factory=dict)
    checks: list[GraderCheck] = Field(default_factory=list)


class RuntimeStatus(BaseModel):
    """Runtime status route payload."""

    model_config = ConfigDict(extra="forbid")

    environment: str = "unified_incident_env"
    progress: UnifiedIncidentState
    grader: GraderReport
