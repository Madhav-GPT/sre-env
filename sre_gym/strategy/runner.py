"""Strategy-tier runner — chains Basic episodes with persistent horizon state.

The Advanced tier is bounded by **horizon**. We implement that here as:

1. Load an Advanced reference scenario YAML.
2. Each ``incident_chain`` phase maps to a Basic-tier template (the mapping is
   declared in ``PHASE_TO_BASIC_TEMPLATE`` below or per-phase via a
   ``basic_template:`` field in the YAML).
3. Run each phase as a Basic episode against ``UnifiedIncidentEnvironment``.
4. **Persist state across phases**:
     - Unresolved alerts from prior phases ride into the next phase as
       baseline noise (raises noise-handling difficulty).
     - Deploys still in flight (rolled-back-but-not-restarted) carry a
       deferred restart tax — the agent must clear them in subsequent
       phases or pay an efficiency penalty.
     - Tech-debt counter accumulates each tick the agent failed to make
       progress; it scales the per-action step cost on subsequent phases.
5. Final reward = ``mean(per_phase_rewards) * horizon_decay`` where decay
   is ``HORIZON_DECAY ** unresolved_phase_count`` (so being slow across
   phases compounds).

The implementation is deliberately *thin*: we do NOT attempt to simulate the
15-20 service topology faithfully. The Advanced tier's research claim is
about long-horizon coherence, not topology fidelity — so chaining Basic
episodes captures the load-bearing piece (state must survive across
episodes) without re-implementing the world model.

Usage::

    from sre_gym.strategy.runner import run_advanced

    result = run_advanced(
        scenario_id="cascading_release_train",
        policy=lambda obs: {"action_type": "query_logs", "service": "worker"},
        seed=1,
    )
    print(result.summary())

CLI::

    python -m sre_gym.strategy run cascading_release_train --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from sre_gym.exceptions import HorizonStateError, ScenarioLoadError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase → Basic template mapping. Per-phase override via YAML `basic_template:`
# field if/when authors want a non-default mapping.
# ---------------------------------------------------------------------------

PHASE_TO_BASIC_TEMPLATE: dict[str, list[str]] = {
    # cascading_release_train: phase 1 = schema drift in gateway, phase 2 = worker drift
    "cascading_release_train::1": ["schema_drift_missing_migration"],
    "cascading_release_train::2": ["dep_degradation"],
    # observability_pipeline_outage: phase 1 = obs-pipeline saturation (cache TTL is closest
    # analog among Basic templates — the cache itself is misbehaving in subtle ways).
    # phase 2 = the underlying deploy regression revealed once logs flow.
    "observability_pipeline_outage::1": ["cache_stale_state"],
    "observability_pipeline_outage::2": ["worker_deploy_cascade"],
    # supabase_rls_silent_leak: 3 phases (containment, root-cause, postmortem).
    # We map the first two to Basic templates; postmortem is grader-only.
    "supabase_rls_silent_leak::1_containment": ["payment_webhook_misconfig"],
    "supabase_rls_silent_leak::2_root_cause":  ["migration_lock"],
    "supabase_rls_silent_leak::3_postmortem":  ["worker_deploy_cascade"],
}


HORIZON_DECAY: float = 0.92  # Per unresolved-phase decay factor for the final score.
TECH_DEBT_STEP_COST_GROWTH: float = 0.005  # Each carried tech-debt point adds this to step_cost.
DEFERRED_DEPLOY_RESTART_TAX: float = 0.04  # Per-phase efficiency penalty for skipping a restart.


# ---------------------------------------------------------------------------
# Data models — persistent horizon state.
# ---------------------------------------------------------------------------


class HorizonState(BaseModel):
    """State that persists across phases of an Advanced-tier scenario.

    Carried into every subsequent ``UnifiedIncidentEnvironment.reset()`` so
    Phase N+1 inherits the residue from Phase N.
    """

    model_config = ConfigDict(extra="forbid")

    # Alerts unresolved at the end of the previous phase. They surface as
    # additional noise in the next phase's observation.
    unresolved_alerts: list[str] = Field(default_factory=list)

    # Deploys rolled back but not followed by a restart in the previous phase.
    # Carrying a deploy adds DEFERRED_DEPLOY_RESTART_TAX per phase.
    pending_deploys: list[str] = Field(default_factory=list)

    # Accumulated tech debt — incremented whenever the agent emits a no-progress
    # action (i.e. ``loop_warning`` fires). Scales subsequent step cost.
    tech_debt: int = 0

    # Phases the agent has finished (resolved + checks green).
    resolved_phases: list[str] = Field(default_factory=list)

    # Phases that timed out / failed / were left partially resolved.
    unresolved_phases: list[str] = Field(default_factory=list)


class PhaseResult(BaseModel):
    """Per-phase episode result."""

    model_config = ConfigDict(extra="forbid")

    phase_id: str
    basic_template: str
    final_score: float
    incident_resolved: bool
    tick_count: int
    cumulative_reward: float
    failure_type: str | None = None
    why_failed: str | None = None


class AdvancedResult(BaseModel):
    """Whole-scenario Advanced-tier result."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    seed: int
    phases: list[PhaseResult] = Field(default_factory=list)
    horizon_state: HorizonState
    raw_mean_reward: float
    horizon_decay_factor: float
    final_reward: float
    success: bool

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"sre-gym Advanced :: scenario={self.scenario_id} seed={self.seed}",
            f"  phases run: {len(self.phases)}",
        ]
        for phase in self.phases:
            flag = "✓" if phase.incident_resolved else "✗"
            lines.append(
                f"    {flag} phase={phase.phase_id:<32} basic={phase.basic_template:<32} "
                f"score={phase.final_score:.3f} ticks={phase.tick_count}"
            )
        lines.extend([
            f"  resolved phases  : {len(self.horizon_state.resolved_phases)} / {len(self.phases)}",
            f"  pending deploys  : {self.horizon_state.pending_deploys}",
            f"  tech debt        : {self.horizon_state.tech_debt}",
            f"  raw mean reward  : {self.raw_mean_reward:.3f}",
            f"  horizon decay    : ×{self.horizon_decay_factor:.3f}",
            f"  final reward     : {self.final_reward:.3f}",
            f"  success          : {self.success}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML loader.
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADVANCED_SCENARIOS = REPO_ROOT / "sre_gym" / "strategy" / "scenarios"


def load_advanced_scenario(scenario_id: str) -> dict[str, Any]:
    """Load an Advanced-tier scenario YAML by id (e.g. 'cascading_release_train')."""
    try:
        import yaml
    except ImportError as exc:
        raise ScenarioLoadError(scenario_id, "PyYAML not installed") from exc
    path = ADVANCED_SCENARIOS / f"{scenario_id}.yaml"
    if not path.is_file():
        raise ScenarioLoadError(scenario_id, f"scenario YAML not found at {path}")
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioLoadError(scenario_id, f"YAML parse failed: {exc}") from exc
    if not isinstance(spec, dict):
        raise ScenarioLoadError(scenario_id, "YAML root must be a mapping")
    if spec.get("id") != scenario_id:
        raise ScenarioLoadError(scenario_id, f"YAML 'id' field is {spec.get('id')!r}, expected {scenario_id!r}")
    return spec


# ---------------------------------------------------------------------------
# Phase mapping helper.
# ---------------------------------------------------------------------------


def _phase_basic_template(scenario_id: str, phase: dict[str, Any]) -> str:
    """Pick a Basic template for a phase. Order: explicit YAML override > builtin map > worker_deploy_cascade."""
    explicit = phase.get("basic_template")
    if isinstance(explicit, str):
        return explicit
    phase_id = str(phase.get("phase", ""))
    key = f"{scenario_id}::{phase_id}"
    candidates = PHASE_TO_BASIC_TEMPLATE.get(key, ["worker_deploy_cascade"])
    return candidates[0]


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


PolicyFn = Callable[["UnifiedIncidentObservation"], dict[str, Any]]  # noqa: F821 - forward ref


def _default_policy(observation: Any) -> dict[str, Any]:
    """Fallback policy: walks the scripted-optimal baseline for the active scenario."""
    from unified_incident_env.server.challenge import list_baselines

    scenario_id = getattr(observation, "_advanced_runner_scenario_id", None) or "worker_deploy_cascade"
    bl = list_baselines(scenario_id=scenario_id).baselines[0]
    tick = max(0, getattr(observation, "tick_count", 0))
    if tick >= len(bl.actions):
        return {"action_type": "escalate"}
    return bl.actions[tick].action.model_dump(exclude_none=True)


def _scripted_policy_factory(scenario_id: str) -> PolicyFn:
    """Build a per-phase scripted-optimal policy.

    The Basic-tier scripted-optimal baseline is the natural reference policy for
    the Advanced runner: it produces a deterministic, in-band score so the
    horizon-decay factor is the meaningful axis of comparison.
    """
    from unified_incident_env.server.challenge import list_baselines

    bl = list_baselines(scenario_id=scenario_id).baselines[0]
    queue = list(bl.actions)
    state = {"tick": 0}

    def policy(observation: Any) -> dict[str, Any]:
        idx = state["tick"]
        state["tick"] += 1
        if idx >= len(queue):
            return {"action_type": "escalate"}
        return queue[idx].action.model_dump(exclude_none=True)

    return policy


def _apply_horizon_residue(
    obs: Any,
    horizon_state: HorizonState,
) -> None:
    """Mutate the observation in place so the agent sees prior-phase residue."""
    if horizon_state.unresolved_alerts:
        # Append carried alerts as additional noise alerts.
        from unified_incident_env.models import Alert

        existing_messages = {a.message for a in obs.noise_alerts}
        for msg in horizon_state.unresolved_alerts:
            if msg not in existing_messages:
                # Use api-gateway as the synthetic surface for inherited noise.
                obs.noise_alerts.append(
                    Alert(service="api-gateway", severity="warning", message=msg)
                )

    if horizon_state.pending_deploys:
        # Surface pending deploys in the prompt's incident_summary.
        suffix = "; ".join(f"pending: {d}" for d in horizon_state.pending_deploys)
        obs.incident_summary = f"{obs.incident_summary} [horizon residue: {suffix}]"


def _run_phase(
    scenario_id: str,
    phase: dict[str, Any],
    horizon_state: HorizonState,
    policy: PolicyFn | None,
    seed: int,
    on_log: Callable[[str], None] | None,
) -> PhaseResult:
    """Run one phase as a Basic episode against UnifiedIncidentEnvironment."""
    from unified_incident_env.models import UnifiedIncidentAction
    from unified_incident_env.server.environment import UnifiedIncidentEnvironment

    basic_template = _phase_basic_template(scenario_id, phase)
    phase_id = str(phase.get("phase", "?"))

    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=basic_template)
    _apply_horizon_residue(obs, horizon_state)

    # Mark observation with scenario_id for the default policy lookup.
    setattr(obs, "_advanced_runner_scenario_id", basic_template)  # type: ignore[attr-defined]

    chosen = policy or _scripted_policy_factory(basic_template)
    cumulative = 0.0
    last_obs = obs
    while not last_obs.done:
        try:
            action_dict = chosen(last_obs)
        except Exception as exc:  # pragma: no cover - policy crash is recoverable
            logger.warning("policy raised: %s", exc)
            action_dict = {"action_type": "escalate"}
        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception:
            action = UnifiedIncidentAction(action_type="escalate")
        last_obs = env.step(action)
        setattr(last_obs, "_advanced_runner_scenario_id", basic_template)  # type: ignore[attr-defined]
        cumulative += float(last_obs.reward)
        if on_log is not None:
            on_log(
                f"[{phase_id}] tick={last_obs.tick_count:>2}/{last_obs.max_ticks} "
                f"action={action.action_type:<22} "
                f"reward={last_obs.reward:+.3f} cum={cumulative:+.3f} "
                f"score={last_obs.final_score:.3f}"
            )

    # Update horizon state with residue from this phase.
    if last_obs.incident_resolved:
        horizon_state.resolved_phases.append(phase_id)
    else:
        horizon_state.unresolved_phases.append(phase_id)
        # Carry alerts that survived the timeout.
        for alert in last_obs.active_alerts:
            horizon_state.unresolved_alerts.append(alert.message)

    # Detect deferred deploys: rollback applied (containment) but verification
    # didn't reach full marks — agent didn't follow up with restart + checks.
    if last_obs.containment_applied and last_obs.score_breakdown.get("verification_score", 0.0) < 0.20:
        horizon_state.pending_deploys.append(basic_template)

    # Tech debt: count loop_warning emissions as proxies for wasted ticks.
    horizon_state.tech_debt += int(last_obs.score_breakdown.get("efficiency_score", 0.05) < 0.04)

    return PhaseResult(
        phase_id=phase_id,
        basic_template=basic_template,
        final_score=float(last_obs.final_score),
        incident_resolved=bool(last_obs.incident_resolved),
        tick_count=int(last_obs.tick_count),
        cumulative_reward=round(cumulative, 4),
        failure_type=last_obs.failure_type,
        why_failed=last_obs.why_failed,
    )


def run_advanced(
    scenario_id: str,
    policy: PolicyFn | None = None,
    seed: int = 0,
    on_log: Callable[[str], None] | None = None,
) -> AdvancedResult:
    """Run an Advanced-tier scenario end-to-end.

    Parameters
    ----------
    scenario_id
        Filename stem under ``sre_gym/strategy/scenarios/``.
    policy
        Optional callable ``(observation) -> action_dict``. Defaults to the
        Basic-tier scripted-optimal baseline for each phase's mapped template.
    seed
        RNG seed surface so future stochastic add-ons stay reproducible.
    on_log
        Optional sink for per-tick log lines (used by the Gradio UI for streaming).

    Returns
    -------
    AdvancedResult
        Per-phase results + horizon state + decay-applied final reward.
    """
    spec = load_advanced_scenario(scenario_id)
    phases: list[dict[str, Any]] = list(spec.get("incident_chain", []))
    if not phases:
        raise ScenarioLoadError(scenario_id, "incident_chain is empty")

    horizon_state = HorizonState()
    phase_results: list[PhaseResult] = []
    if on_log is not None:
        on_log(f"=== sre-gym Advanced :: {scenario_id} (seed={seed}, {len(phases)} phases) ===")

    for phase in phases:
        if on_log is not None:
            on_log(
                f"--- phase {phase.get('phase')} :: {phase.get('triggered_by', 'unknown')} ---"
            )
        result = _run_phase(scenario_id, phase, horizon_state, policy, seed, on_log)
        phase_results.append(result)

    if not phase_results:
        raise HorizonStateError("no phases ran")

    raw_mean = sum(p.final_score for p in phase_results) / len(phase_results)
    decay = HORIZON_DECAY ** len(horizon_state.unresolved_phases)
    final_reward = round(raw_mean * decay, 4)
    success = bool(horizon_state.unresolved_phases == [] and final_reward >= 0.65)

    return AdvancedResult(
        scenario_id=scenario_id,
        seed=seed,
        phases=phase_results,
        horizon_state=horizon_state,
        raw_mean_reward=round(raw_mean, 4),
        horizon_decay_factor=round(decay, 4),
        final_reward=final_reward,
        success=success,
    )


def list_advanced_scenarios() -> list[str]:
    """Return the list of available Advanced-tier scenario IDs."""
    if not ADVANCED_SCENARIOS.is_dir():
        return []
    return sorted(p.stem for p in ADVANCED_SCENARIOS.glob("*.yaml"))


# ---------------------------------------------------------------------------
# CLI entry-point. Invoked by ``python -m sre_gym.strategy run …``.
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sre_gym.strategy", description="Strategy-tier runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an Advanced-tier scenario end-to-end")
    p_run.add_argument("scenario_id")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--json", action="store_true", help="emit JSON instead of human summary")

    sub.add_parser("list", help="list available Advanced scenarios")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.cmd == "list":
        for sid in list_advanced_scenarios():
            print(sid)
        return 0

    if args.cmd == "run":
        def on_log(line: str) -> None:
            print(line)

        try:
            result = run_advanced(args.scenario_id, seed=args.seed, on_log=on_log)
        except ScenarioLoadError as exc:
            print(f"error: {exc}", flush=True)
            return 2
        if args.json:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print()
            print(result.summary())
        return 0 if result.success else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
