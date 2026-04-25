"""Basic-tier scenario runner.

Wraps ``UnifiedIncidentEnvironment`` in the same ``run() -> Result`` shape the
Advanced and Max runners use, so the Gradio UI / CLI can drive all three tiers
with the same calling convention.

Single-episode rollout. Defaults to the scripted-optimal baseline policy if no
``policy`` is supplied — useful for the Gradio UI's "ship a tracable demo
without an LLM" path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

PolicyFn = Callable[[Any], dict[str, Any]]


class BasicResult(BaseModel):
    """Single-episode Basic-tier rollout result."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    seed: int
    incident_resolved: bool
    final_score: float
    tick_count: int
    cumulative_reward: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    failure_type: str | None = None
    why_failed: str | None = None

    def summary(self) -> str:
        lines = [
            f"sre-gym Basic :: scenario={self.scenario_id} seed={self.seed}",
            f"  resolved   : {self.incident_resolved}",
            f"  ticks      : {self.tick_count}",
            f"  cum reward : {self.cumulative_reward:+.3f}",
            f"  final score: {self.final_score:.3f}",
        ]
        if self.failure_type:
            lines.append(f"  failure    : {self.failure_type} — {self.why_failed}")
        return "\n".join(lines)


def _scripted_policy(scenario_id: str) -> PolicyFn:
    """Closure that walks the scripted-optimal baseline action-by-action."""
    from unified_incident_env.server.challenge import list_baselines

    bl = list_baselines(scenario_id=scenario_id).baselines[0]
    queue = list(bl.actions)
    state = {"i": 0}

    def policy(_obs: Any) -> dict[str, Any]:
        idx = state["i"]
        state["i"] += 1
        if idx >= len(queue):
            return {"action_type": "escalate"}
        return queue[idx].action.model_dump(exclude_none=True)

    return policy


def run_basic(
    scenario_id: str,
    *,
    policy: PolicyFn | None = None,
    seed: int = 0,
    on_log: Callable[[str], None] | None = None,
    max_ticks: int | None = None,
) -> BasicResult:
    """Run a Basic-tier scenario end-to-end.

    Parameters
    ----------
    scenario_id
        Any catalogue ID (template or procgen variant, e.g. ``memory_leak_oom__p02``).
    policy
        Callable ``(observation) -> action_dict``. Defaults to the scripted-optimal
        baseline for the scenario's template.
    seed
        Reserved for future stochastic behaviour; currently informational.
    on_log
        Optional sink for per-tick log lines (used by the Gradio UI for streaming).
    max_ticks
        Optional override for the scenario's natural ``max_ticks``. Useful for
        smoke tests; production use should leave it ``None``.
    """
    from unified_incident_env.models import UnifiedIncidentAction
    from unified_incident_env.server.environment import UnifiedIncidentEnvironment

    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    cumulative = 0.0

    chosen = policy or _scripted_policy(scenario_id)

    if on_log is not None:
        on_log(f"=== sre-gym Basic :: {scenario_id} (seed={seed}) ===")

    while not obs.done:
        if max_ticks is not None and obs.tick_count >= max_ticks:
            break
        try:
            action_dict = chosen(obs)
        except Exception as exc:  # pragma: no cover
            logger.warning("policy raised: %s", exc)
            action_dict = {"action_type": "escalate"}
        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception:
            action = UnifiedIncidentAction(action_type="escalate")
        obs = env.step(action)
        cumulative += float(obs.reward)
        if on_log is not None:
            args = {k: v for k, v in action.model_dump(exclude_none=True).items() if k != "action_type"}
            on_log(
                f"tick={obs.tick_count:>2}/{obs.max_ticks} "
                f"action={action.action_type:<22} "
                f"args={json.dumps(args, separators=(',', ':')):<48} "
                f"reward={obs.reward:+.3f} cum={cumulative:+.3f} "
                f"score={obs.final_score:.3f}"
            )

    if on_log is not None:
        on_log(
            f"DONE resolved={obs.incident_resolved} ticks={obs.tick_count} "
            f"final_score={obs.final_score:.3f}"
        )

    return BasicResult(
        scenario_id=scenario_id,
        seed=seed,
        incident_resolved=bool(obs.incident_resolved),
        final_score=float(obs.final_score),
        tick_count=int(obs.tick_count),
        cumulative_reward=round(cumulative, 4),
        score_breakdown=dict(obs.score_breakdown),
        failure_type=obs.failure_type,
        why_failed=obs.why_failed,
    )
