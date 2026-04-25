"""5-component rubric grader for the sre-gym Triage tier.

The earlier 7-dim grader (recovery / containment / verification / impact /
efficiency / speed_bonus / noise_handling) summed to 0.85 and flattened the
GRPO advantage signal. This grader collapses to 5 components that sum to
exactly 1.0 and surfaces an explicit anti-cheat dimension:

    outcome          0.45    root-cause action correct + recovery confirmed
    action_validity  0.20    fraction of actions that are valid + correctly typed
    format           0.10    submit_hypothesis was called before declare_resolved
    anticheat        0.15    declare_resolved blocked unless ≥1 query-action ran
    efficiency       0.10    exp(-steps_used / steps_optimal_for_template)
                     ----
    composite        1.00

The ``unsafe_action_penalty`` referenced elsewhere is an action-level *step
penalty* applied inside ``UnifiedIncidentEnvironment.step``; it is NOT a
rubric component. See ``docs/REWARD_DESIGN.md`` for the full rationale.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import GraderCheck, GraderReport

MIN_PUBLIC_SCORE = 0.01
MAX_PUBLIC_SCORE = 0.99


class RubricScore(BaseModel):
    """The 5-component rubric. Weights are baked into ``composite``."""

    model_config = ConfigDict(extra="forbid")

    outcome: float = Field(..., ge=0.0, le=1.0)
    action_validity: float = Field(..., ge=0.0, le=1.0)
    format: float = Field(..., ge=0.0, le=1.0)
    anticheat: float = Field(..., ge=0.0, le=1.0)
    efficiency: float = Field(..., ge=0.0, le=1.0)

    @property
    def composite(self) -> float:
        return round(
            0.45 * self.outcome
            + 0.20 * self.action_validity
            + 0.10 * self.format
            + 0.15 * self.anticheat
            + 0.10 * self.efficiency,
            4,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "outcome": round(self.outcome, 4),
            "action_validity": round(self.action_validity, 4),
            "format": round(self.format, 4),
            "anticheat": round(self.anticheat, 4),
            "efficiency": round(self.efficiency, 4),
        }


def _strict_public_score(score: float) -> float:
    return round(min(MAX_PUBLIC_SCORE, max(MIN_PUBLIC_SCORE, score)), 4)


# Action families used by the anticheat dimension. Anything not in this set
# does NOT count as "earning" the anticheat dimension — even run_check and
# submit_hypothesis don't, because they're not evidence-gathering on the
# four golden signals.
_QUERY_ACTIONS = {"query_logs", "query_metrics", "query_dependencies", "query_deploys"}


class UnifiedIncidentGrader:
    """Computes the 5-component rubric from the env's per-step state.

    The env is responsible for emitting these state fields — see
    ``unified_incident_env/server/environment.py``:

      action_counts                — dict[action_type, int]
      invalid_action_count         — int (unsupported / unknown actions)
      query_actions_before_resolved — int  (sum of query_* before declare_resolved)
      hypothesis_submitted_at_tick — int | None
      declare_resolved_called_at_tick — int | None
      cause_removed                — bool
      hypothesis_root_cause_correct — bool
      step_count                   — int
      current_tick                 — int
      checks                       — list[CheckResult]
      incident_resolved            — bool
    """

    def compute_rubric(
        self,
        state: dict[str, Any],
        scenario: dict[str, Any],
    ) -> RubricScore:
        # ---- outcome ----
        cause_removed = bool(state.get("cause_removed"))
        checks = {item.get("name"): bool(item.get("passed")) for item in state.get("checks", [])}
        end_to_end_passed = checks.get("end_to_end", False)
        hypothesis_correct = bool(state.get("hypothesis_root_cause_correct"))
        if cause_removed and end_to_end_passed:
            outcome = 1.0
        elif hypothesis_correct and not cause_removed:
            outcome = 0.5
        else:
            outcome = 0.0

        # ---- action_validity ----
        step_count = int(state.get("step_count", 0))
        invalid = int(state.get("invalid_action_count", 0))
        action_validity = 0.0
        if step_count > 0:
            action_validity = max(0.0, (step_count - invalid) / step_count)

        # ---- format ----
        hypo_at = state.get("hypothesis_submitted_at_tick")
        resolve_at = state.get("declare_resolved_called_at_tick")
        if hypo_at is None:
            fmt = 0.0
        elif resolve_at is None:
            # Hypothesis was submitted, declare_resolved never called — that's
            # still 1.0 for format because the order constraint is satisfied
            # vacuously. (The outcome dimension penalises non-resolution.)
            fmt = 1.0
        else:
            fmt = 1.0 if hypo_at < resolve_at else 0.0

        # ---- anticheat ----
        # 1.0 iff at least one query_* action ran before the FIRST
        # declare_resolved attempt. Blocks shortcut-resolve attempts.
        queries_before = int(state.get("query_actions_before_resolved", 0))
        anticheat = 1.0 if queries_before >= 1 else 0.0

        # ---- efficiency ----
        steps_used = int(state.get("current_tick", 0))
        steps_optimal = max(1, int(scenario.get("optimal_ticks", 10)))
        efficiency = min(1.0, math.exp(-steps_used / steps_optimal))

        return RubricScore(
            outcome=outcome,
            action_validity=action_validity,
            format=fmt,
            anticheat=anticheat,
            efficiency=efficiency,
        )

    def compute_breakdown(
        self,
        state: dict[str, Any],
        scenario: dict[str, Any],
    ) -> dict[str, float]:
        """Backwards-compatible breakdown dict used by the env's observation.

        The new keys (outcome/action_validity/format/anticheat/efficiency)
        are the canonical surface; the legacy 7-dim keys are retained as
        zeroed compatibility values so any caller that still reads the old
        shape doesn't crash. ``final_score`` is the composite under the new
        rubric.
        """
        rubric = self.compute_rubric(state, scenario)
        composite = rubric.composite
        out = rubric.as_dict()
        out.update({
            "final_score": _strict_public_score(composite),
            # Legacy keys kept at 0.0 so existing callers / tests can still read them.
            "recovery_score": 0.0,
            "containment_score": 0.0,
            "verification_score": 0.0,
            "impact_score": 0.0,
            "efficiency_score": 0.0,
            "speed_bonus": 0.0,
            "noise_handling_score": 0.0,
        })
        return out

    def build_report(self, state: dict[str, Any], scenario: dict[str, Any]) -> GraderReport:
        rubric = self.compute_rubric(state, scenario)
        breakdown = self.compute_breakdown(state, scenario)
        passed = rubric.outcome >= 1.0
        report_checks = [
            GraderCheck(
                name="outcome",
                passed=rubric.outcome >= 1.0,
                detail=(
                    "Root cause was correctly remediated and verified."
                    if rubric.outcome >= 1.0
                    else (
                        "Root cause identified but no successful remediation."
                        if rubric.outcome >= 0.5
                        else "Root cause not removed; recovery not verified."
                    )
                ),
                weight=0.45,
            ),
            GraderCheck(
                name="action_validity",
                passed=rubric.action_validity >= 0.9,
                detail=(
                    f"{rubric.action_validity:.2%} of actions were structurally valid."
                ),
                weight=0.20,
            ),
            GraderCheck(
                name="format",
                passed=rubric.format >= 1.0,
                detail=(
                    "submit_hypothesis was called before declare_resolved."
                    if rubric.format >= 1.0
                    else "submit_hypothesis must be called before declare_resolved."
                ),
                weight=0.10,
            ),
            GraderCheck(
                name="anticheat",
                passed=rubric.anticheat >= 1.0,
                detail=(
                    "At least one query_* action ran before declare_resolved."
                    if rubric.anticheat >= 1.0
                    else "declare_resolved without prior evidence-gathering — blocked."
                ),
                weight=0.15,
            ),
            GraderCheck(
                name="efficiency",
                passed=rubric.efficiency >= 0.5,
                detail=(
                    f"exp(-{state.get('current_tick', 0)} / "
                    f"{scenario.get('optimal_ticks', 10)}) = {rubric.efficiency:.3f}"
                ),
                weight=0.10,
            ),
        ]
        return GraderReport(
            scenario_id=scenario["id"],
            passed=passed,
            score=breakdown["final_score"],
            message=(
                "Outcome dimension cleared: incident remediated and verified."
                if passed
                else "Outcome dimension not cleared yet."
            ),
            breakdown=breakdown,
            checks=report_checks,
        )
