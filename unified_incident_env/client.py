"""Typed OpenEnv client for the unified incident environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import UnifiedIncidentAction, UnifiedIncidentObservation, UnifiedIncidentState


class UnifiedIncidentEnv(
    EnvClient[UnifiedIncidentAction, UnifiedIncidentObservation, UnifiedIncidentState]
):
    """Typed client wrapper around the OpenEnv HTTP API."""

    DEFAULT_BASE_URL = "http://127.0.0.1:8000"

    def _step_payload(self, action: UnifiedIncidentAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[UnifiedIncidentObservation]:
        observation_data = dict(payload.get("observation", {}))
        observation_data.setdefault("reward", payload.get("reward", 0.0))
        observation_data.setdefault("done", payload.get("done", False))
        observation = UnifiedIncidentObservation.model_validate(observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> UnifiedIncidentState:
        return UnifiedIncidentState.model_validate(payload)
