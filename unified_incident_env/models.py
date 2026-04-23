"""Typed models for the honest narrow incident-remediation environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

ActionType = Literal[
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
Difficulty = Literal["easy"]
MetricName = Literal["cpu", "error_rate", "latency"]
ServiceName = Literal["api-gateway", "cache", "database", "worker"]
ServiceStatus = Literal["healthy", "degraded", "crashed", "isolated"]
WorkflowStage = Literal["triage", "mitigation", "validation", "resolved"]
CheckName = Literal["database_recovery", "end_to_end"]
RootCauseType = Literal[
    "bad_worker_deploy",
    "database_only_failure",
    "api_gateway_fault",
]
RecommendedActionType = Literal[
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
]


class PostmortemPayload(BaseModel):
    """Deprecated compatibility shell for the removed v1 postmortem action."""

    model_config = ConfigDict(extra="forbid")

    root_cause: str = ""
    attack_vector: str = ""
    timeline: list[str] = Field(default_factory=list)
    remediation_steps: list[str] = Field(default_factory=list)
    prevention_steps: list[str] = Field(default_factory=list)


class SecurityContext(BaseModel):
    """Deprecated compatibility shell for the removed v1 security subquest state."""

    model_config = ConfigDict(extra="forbid")

    code_visible: bool = False
    selected_vulnerability: str | None = None
    selected_patch: str | None = None
    exploit_blocked: bool | None = None
    functionality_preserved: bool | None = None


class HypothesisPayload(BaseModel):
    """Structured hypothesis submitted by the agent."""

    model_config = ConfigDict(extra="forbid")

    root_cause: RootCauseType
    affected_services: list[ServiceName] = Field(default_factory=list, min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_next_action: RecommendedActionType


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


class CheckResult(BaseModel):
    """Result of a verification check."""

    model_config = ConfigDict(extra="forbid")

    name: CheckName
    passed: bool
    detail: str


class UnifiedIncidentAction(Action):
    """One structured environment action."""

    model_config = ConfigDict(extra="ignore")

    action_type: ActionType
    service: ServiceName | None = None
    metric: MetricName | None = None
    check_name: CheckName | None = None
    hypothesis: HypothesisPayload | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "UnifiedIncidentAction":
        if self.action_type in {
            "query_logs",
            "query_dependencies",
            "query_deploys",
            "rollback_deploy",
            "restart_service",
            "isolate_service",
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
                    {"action_type": self.action_type},
                )
            if not self.metric:
                raise PydanticCustomError(
                    "missing_metric",
                    "metric is required for {action_type}",
                    {"action_type": self.action_type},
                )
        if self.action_type == "run_check" and not self.check_name:
            raise PydanticCustomError(
                "missing_check_name",
                "check_name is required for {action_type}",
                {"action_type": self.action_type},
            )
        if self.action_type == "submit_hypothesis" and self.hypothesis is None:
            raise PydanticCustomError(
                "missing_hypothesis",
                "hypothesis is required for {action_type}",
                {"action_type": self.action_type},
            )
        return self


class UnifiedIncidentObservation(Observation):
    """Observation returned after reset and each step."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    incident_summary: str
    tick_count: int
    max_ticks: int
    difficulty: Difficulty
    workflow_stage: WorkflowStage
    active_alerts: list[Alert] = Field(default_factory=list)
    service_health: dict[str, ServiceHealth] = Field(default_factory=dict)
    discovered_evidence: list[str] = Field(default_factory=list)
    recent_deploys: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    user_impact: float = Field(ge=0.0, le=1.0)
    slo_burn_rate: float = Field(ge=0.0, le=1.0)
    incident_resolved: bool = False
    containment_applied: bool = False
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
    security_subquest_status: str | None = None
    security_context: dict[str, Any] = Field(default_factory=dict)
    final_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
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
    recent_deploys: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    user_impact: float = Field(ge=0.0, le=1.0)
    slo_burn_rate: float = Field(ge=0.0, le=1.0)
    incident_resolved: bool = False
    containment_applied: bool = False
    allowed_actions: list[str] = Field(default_factory=list)
    required_fields_by_action: dict[str, list[str]] = Field(default_factory=dict)
    valid_action_example: dict[str, Any] | None = None
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    final_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    wasteful_ticks: int = 0
    last_action_result: str = ""
    failure_type: str | None = None
    why_failed: str | None = None


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
