"""Typed records for trainer requests, trajectories, datasets, and sessions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FailureBucketName = Literal["schema", "policy", "looping", "reasoning"]
ParseStatus = Literal["ok", "repaired", "teacher_override", "invalid_json", "invalid_action"]
UpdateBackendName = Literal["noop", "external_command", "openai_finetune"]
UpdateStatus = Literal["completed", "noop", "failed"]
SessionPhaseName = Literal["probe", "first_correction", "workflow_correction", "final_evaluation"]
TrainerMode = Literal["strict", "lenient"]
RuntimeMode = Literal["competition", "research"]
StructuredMode = Literal[
    "plain_json",
    "response_format_json",
    "tool_calling",
    "backend_adaptive",
]
SFTSource = Literal[
    "baseline",
    "replay",
    "schema_repair",
    "next_action",
    "recovery",
]


class ModelRequest(BaseModel):
    """One backend completion request."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    system_prompt: str
    user_prompt: str
    structured_mode: StructuredMode = "backend_adaptive"
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int = 220


class ModelResponse(BaseModel):
    """One backend completion response."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    latency_s: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    """Result of converting raw model text to an action."""

    model_config = ConfigDict(extra="forbid")

    parse_status: ParseStatus
    cleaned_action: dict[str, Any] | None = None
    error: str | None = None
    repair_labels: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    """One recorded environment step or parse failure."""

    model_config = ConfigDict(extra="forbid")

    episode_id: int | None = None
    scenario_id: str | None = None
    step_index: int
    tick: int
    workflow_stage: str
    observation: dict[str, Any] = Field(default_factory=dict)
    prompt_text: str
    raw_model_output: str
    parse_status: ParseStatus
    normalization_applied: list[str] = Field(default_factory=list)
    cleaned_action: dict[str, Any] | None = None
    teacher_action: dict[str, Any] | None = None
    reward: float | None = None
    cumulative_score: float | None = None
    final_score_after_step: float | None = None
    done: bool = False
    next_prompt_text: str | None = None
    structured_mode_used: StructuredMode = "plain_json"
    repair_retry_used: bool = False
    teacher_override_used: bool = False
    failure_reason: str | None = None
    failure_type: str | None = None


class FailureBucketEntry(BaseModel):
    """One classified failure instance."""

    model_config = ConfigDict(extra="forbid")

    episode_id: int
    scenario_id: str
    step_index: int | None = None
    bucket: FailureBucketName
    failure_type: str
    detail: str


class FailureAnalysisReport(BaseModel):
    """Failure analysis for one episode or one block."""

    model_config = ConfigDict(extra="forbid")

    episode_ids: list[int] = Field(default_factory=list)
    scenario_ids: list[str] = Field(default_factory=list)
    entries: list[FailureBucketEntry] = Field(default_factory=list)
    schema_failures: list[str] = Field(default_factory=list)
    policy_failures: list[str] = Field(default_factory=list)
    looping_failures: list[str] = Field(default_factory=list)
    reasoning_failures: list[str] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class EpisodeSummaryRecord(BaseModel):
    """Compact episode-level summary used in session reports."""

    model_config = ConfigDict(extra="forbid")

    episode_id: int
    run_id: str
    scenario_id: str
    difficulty: str
    model_name: str
    model_version: str | None = None
    mode: TrainerMode
    steps: int
    success: bool
    final_score: float
    schema_failures: int = 0
    json_valid_steps: int = 0
    strict_schema_valid_steps: int = 0
    teacher_override_count: int = 0
    repair_retry_count: int = 0
    policy_failures: list[str] = Field(default_factory=list)
    looping_failures: list[str] = Field(default_factory=list)
    reasoning_failures: list[str] = Field(default_factory=list)
    security_subquest_completed: bool = False
    postmortem_completed: bool = False
    stopped_reason: str | None = None
    elapsed_s: float


class EpisodeRecord(BaseModel):
    """One full episode trajectory."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    episode_id: int | None = None
    scenario_id: str
    difficulty: str
    model_name: str
    model_version: str | None = None
    mode: TrainerMode
    success: bool
    final_score: float
    steps: int
    failure_reason: str | None = None
    elapsed_s: float
    schema_failures: int = 0
    json_valid_steps: int = 0
    strict_schema_valid_steps: int = 0
    teacher_override_count: int = 0
    repair_retry_count: int = 0
    policy_failures: list[str] = Field(default_factory=list)
    looping_failures: list[str] = Field(default_factory=list)
    reasoning_failures: list[str] = Field(default_factory=list)
    security_subquest_completed: bool = False
    postmortem_completed: bool = False
    stopped_reason: str | None = None
    step_records: list[StepRecord] = Field(default_factory=list)


class EvalScenarioResult(BaseModel):
    """Summary for one model/scenario pair."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    scenario_id: str
    success: bool
    final_score: float
    failure_reason: str | None = None
    schema_failure: bool = False
    elapsed_s: float


class EvalSummary(BaseModel):
    """Aggregate evaluation report."""

    model_config = ConfigDict(extra="forbid")

    mode: TrainerMode
    results: list[EvalScenarioResult] = Field(default_factory=list)
    success_rate: float = 0.0
    avg_score: float = 0.0
    schema_failure_rate: float = 0.0
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_scenario: dict[str, dict[str, float]] = Field(default_factory=dict)


class SFTRecord(BaseModel):
    """One supervised fine-tuning row."""

    model_config = ConfigDict(extra="forbid")

    source: SFTSource
    scenario_id: str
    tick: int
    messages: list[dict[str, str]]
    target_action: dict[str, Any]
    student_action: dict[str, Any] | None = None
    parse_status: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateRequest(BaseModel):
    """One request to update a model between episode blocks."""

    model_config = ConfigDict(extra="forbid")

    update_index: int
    phase_name: SessionPhaseName
    episodes_used: list[int] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    model_before: str
    output_dir: str
    runtime_mode: RuntimeMode = "competition"
    command_template: str | None = None
    training_file: str | None = None
    suffix: str | None = None


class UpdateResult(BaseModel):
    """Result of one update stage."""

    model_config = ConfigDict(extra="forbid")

    update_index: int
    phase_name: SessionPhaseName
    updater_backend: UpdateBackendName
    model_before: str
    model_after: str
    status: UpdateStatus
    episodes_used: list[int] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    notes: str | None = None


class SessionPhaseConfig(BaseModel):
    """One configured phase in the ten-episode schedule."""

    model_config = ConfigDict(extra="forbid")

    phase_name: SessionPhaseName
    episode_ids: list[int]
    update_after: bool = False
    update_index: int | None = None


class SessionConfig(BaseModel):
    """Configuration for one ten-episode session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    model_name: str
    initial_model_version: str
    runtime_mode: RuntimeMode = "competition"
    collection_mode: TrainerMode = "lenient"
    final_eval_mode: TrainerMode = "strict"
    log_rendered_prompts: bool = False
    base_url: str | None = None
    api_base_url: str
    api_key: str
    output_root: str
    scenario_schedule: list[str]
    phases: list[SessionPhaseConfig]
    updater_backend: UpdateBackendName = "noop"
    updater_command_template: str | None = None


class SessionPhaseReport(BaseModel):
    """Aggregate report for one phase."""

    model_config = ConfigDict(extra="forbid")

    phase_name: SessionPhaseName
    episode_ids: list[int]
    avg_score: float
    success_rate: float
    schema_failures: int
    loop_failures: int = 0
    updates_applied: list[int] = Field(default_factory=list)


class PhaseDeltaRecord(BaseModel):
    """Delta from the previous phase for key session metrics."""

    model_config = ConfigDict(extra="forbid")

    phase_name: SessionPhaseName
    score_delta: float
    schema_failure_delta: float
    loop_failure_delta: float
    success_delta: float


class SessionReport(BaseModel):
    """Full ten-episode training/eval session report."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    model_name: str
    runtime_mode: RuntimeMode
    output_dir: str
    episode_summaries: list[EpisodeSummaryRecord] = Field(default_factory=list)
    updates: list[UpdateResult] = Field(default_factory=list)
    phase_reports: list[SessionPhaseReport] = Field(default_factory=list)
    phase_deltas: list[PhaseDeltaRecord] = Field(default_factory=list)
    improvement_metrics: dict[str, float] = Field(default_factory=dict)
    correction_memory_stats: dict[str, Any] = Field(default_factory=dict)
