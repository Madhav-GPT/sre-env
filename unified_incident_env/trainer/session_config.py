"""Configuration helpers for the ten-episode session loop."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from ..server.challenge import SCENARIOS
from .types import RuntimeMode, SessionConfig, SessionPhaseConfig

DEFAULT_UPDATE_EPISODES = (2, 5, 8)


def default_scenario_schedule() -> list[str]:
    """Return a deterministic ten-episode round-robin schedule."""
    ordered = list(SCENARIOS)
    schedule: list[str] = []
    while len(schedule) < 10:
        schedule.extend(ordered)
    return schedule[:10]


def default_phases() -> list[SessionPhaseConfig]:
    """Return the fixed four-phase ten-episode schedule."""
    return [
        SessionPhaseConfig(
            phase_name="probe",
            episode_ids=[1, 2],
            update_after=True,
            update_index=1,
        ),
        SessionPhaseConfig(
            phase_name="first_correction",
            episode_ids=[3, 4, 5],
            update_after=True,
            update_index=2,
        ),
        SessionPhaseConfig(
            phase_name="workflow_correction",
            episode_ids=[6, 7, 8],
            update_after=True,
            update_index=3,
        ),
        SessionPhaseConfig(
            phase_name="final_evaluation",
            episode_ids=[9, 10],
            update_after=False,
            update_index=None,
        ),
    ]


def make_session_config(
    *,
    model_name: str,
    output_root: str | Path,
    base_url: str | None,
    api_base_url: str,
    api_key: str,
    runtime_mode: RuntimeMode = "competition",
    collection_mode: str = "lenient",
    final_eval_mode: str = "strict",
    updater_backend: str = "external_command",
    updater_command_template: str | None = None,
    scenario_schedule: list[str] | None = None,
    phases: list[SessionPhaseConfig] | None = None,
    log_rendered_prompts: bool | None = None,
) -> SessionConfig:
    """Create one deterministic session config."""
    session_id = uuid.uuid4().hex[:12]
    output_root_path = Path(output_root)
    output_dir = output_root_path / f"session-{session_id}"
    schedule = scenario_schedule or default_scenario_schedule()
    if log_rendered_prompts is None:
        log_rendered_prompts = runtime_mode == "research"
    return SessionConfig(
        session_id=session_id,
        model_name=model_name,
        initial_model_version=model_name,
        runtime_mode=runtime_mode,
        collection_mode=collection_mode,
        final_eval_mode=final_eval_mode,
        log_rendered_prompts=log_rendered_prompts,
        base_url=base_url,
        api_base_url=api_base_url,
        api_key=api_key,
        output_root=str(output_dir),
        scenario_schedule=schedule,
        phases=phases or default_phases(),
        updater_backend=updater_backend,
        updater_command_template=updater_command_template,
    )


def make_session_config_from_env(
    *,
    model_name: str = "qwen2.5:1.5b",
    output_root: str | Path = "outputs/trainer",
    base_url: str | None = None,
    runtime_mode: RuntimeMode = "competition",
    collection_mode: str = "lenient",
    final_eval_mode: str = "strict",
    updater_backend: str = "external_command",
    updater_command_template: str | None = None,
    phases: list[SessionPhaseConfig] | None = None,
    log_rendered_prompts: bool | None = None,
) -> SessionConfig:
    """Create a session config using the common environment variables."""
    return make_session_config(
        model_name=model_name,
        output_root=output_root,
        base_url=base_url,
        runtime_mode=runtime_mode,
        api_base_url=os.environ.get("API_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
        or "local",
        collection_mode=collection_mode,
        final_eval_mode=final_eval_mode,
        updater_backend=updater_backend,
        updater_command_template=updater_command_template,
        phases=phases,
        log_rendered_prompts=log_rendered_prompts,
    )
