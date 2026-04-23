"""Deterministic public scoring for the honest narrow incident-remediation environment."""

from __future__ import annotations

from typing import Any

from ..models import GraderCheck, GraderReport

MIN_PUBLIC_SCORE = 0.01
MAX_PUBLIC_SCORE = 0.99


def _strict_public_score(score: float) -> float:
    return round(min(MAX_PUBLIC_SCORE, max(MIN_PUBLIC_SCORE, score)), 4)


def _service_score(status: str) -> float:
    return {
        "healthy": 1.0,
        "degraded": 0.4,
        "crashed": 0.0,
        "isolated": 0.2,
    }.get(status, 0.0)


class UnifiedIncidentGrader:
    """Deterministic scorer focused on executed effects, not scripted clues.

    Hardened schedule (post Track-A headroom patch):

    - recovery       0.00 – 0.25
    - containment    0.00 – 0.15
    - verification   0.00 – 0.20
    - impact         0.00 – 0.05
    - efficiency     0.00 – 0.05
    - speed_bonus    0.00 – 0.10    (positive only when faster than optimal)
    - noise_handling 0.00 – 0.05    (penalizes querying noise services)

    Scripted deterministic baseline (which matches optimal_ticks exactly and
    avoids noise queries) caps at ~0.70. Headroom 0.70 → 0.85 is reachable only
    by an agent that (a) is strictly faster than optimal and (b) touches zero
    noise services. That's the training target.
    """

    def compute_breakdown(
        self,
        state: dict[str, Any],
        scenario: dict[str, Any],
    ) -> dict[str, float]:
        services = state.get("service_health", {})
        weights = scenario["critical_service_weights"]
        recovery_raw = sum(
            weights.get(service, 0.0) * _service_score((services.get(service) or {}).get("status", "crashed"))
            for service in weights
        )
        recovery_score = round(0.25 * recovery_raw, 4)

        contained = bool(state.get("containment_applied"))
        rollback_target = scenario.get("remediation_recipe", {}).get("rollback_target")
        rollback_service_healthy = bool(
            rollback_target and (services.get(rollback_target) or {}).get("status") == "healthy"
        )
        if contained and rollback_service_healthy:
            containment_score = 0.15
        elif contained:
            containment_score = 0.10
        else:
            containment_score = 0.0

        checks = {item.get("name"): bool(item.get("passed")) for item in state.get("checks", [])}
        verification_score = 0.0
        if checks.get("database_recovery"):
            verification_score += 0.08
        if checks.get("end_to_end"):
            verification_score += 0.12

        user_impact = float(state.get("user_impact", 1.0))
        impact_score = round(max(0.0, 0.05 * (1.0 - user_impact)), 4)

        wasteful_ticks = int(state.get("wasteful_ticks", 0))
        efficiency_score = round(max(0.0, 0.05 - (0.005 * wasteful_ticks)), 4)

        # speed_bonus: fully earned only if the agent finishes well under optimal_ticks.
        optimal_ticks = int(scenario.get("optimal_ticks", 10))
        current_tick = int(state.get("current_tick", 0))
        incident_resolved = bool(state.get("incident_resolved"))
        if incident_resolved and current_tick > 0 and current_tick < optimal_ticks:
            speed_bonus = round(0.10 * (optimal_ticks - current_tick) / optimal_ticks, 4)
        elif incident_resolved and current_tick == optimal_ticks:
            speed_bonus = 0.0
        else:
            speed_bonus = 0.0

        # noise_handling: deduct per query against a noise service, up to the cap of 0.05.
        noise_services = set(scenario.get("difficulty_knobs", {}).get("noise_services", []))
        noise_queries = int(state.get("noise_queries", 0))
        if noise_services:
            noise_handling_score = round(max(0.0, 0.05 - 0.015 * noise_queries), 4)
        else:
            noise_handling_score = 0.0

        final_score = _strict_public_score(
            recovery_score
            + containment_score
            + verification_score
            + impact_score
            + efficiency_score
            + speed_bonus
            + noise_handling_score
        )

        return {
            "recovery_score": recovery_score,
            "containment_score": round(containment_score, 4),
            "verification_score": round(verification_score, 4),
            "impact_score": impact_score,
            "efficiency_score": efficiency_score,
            "speed_bonus": speed_bonus,
            "noise_handling_score": noise_handling_score,
            "final_score": final_score,
        }

    def build_report(self, state: dict[str, Any], scenario: dict[str, Any]) -> GraderReport:
        breakdown = self.compute_breakdown(state, scenario)
        checks = {item.get("name"): bool(item.get("passed")) for item in state.get("checks", [])}
        passed = bool(
            state.get("incident_resolved")
            and checks.get("database_recovery")
            and checks.get("end_to_end")
        )
        report_checks = [
            GraderCheck(
                name="root_cause_removed",
                passed=bool(state.get("containment_applied")),
                detail=(
                    "The root cause has been safely contained or removed."
                    if state.get("containment_applied")
                    else "The root cause is still active or only partially contained."
                ),
                weight=0.20,
            ),
            GraderCheck(
                name="database_recovery",
                passed=checks.get("database_recovery", False),
                detail=(
                    "The database recovery check passed."
                    if checks.get("database_recovery")
                    else "The database recovery check has not passed yet."
                ),
                weight=0.15,
            ),
            GraderCheck(
                name="end_to_end_check",
                passed=checks.get("end_to_end", False),
                detail=(
                    "The end-to-end service check passed."
                    if checks.get("end_to_end")
                    else "The end-to-end service check has not passed yet."
                ),
                weight=0.20,
            ),
            GraderCheck(
                name="critical_services_recovered",
                passed=breakdown["recovery_score"] >= 0.20,
                detail=(
                    "Critical-path services are recovered."
                    if breakdown["recovery_score"] >= 0.20
                    else "Critical-path services are still degraded or crashed."
                ),
                weight=0.20,
            ),
            GraderCheck(
                name="declare_resolved",
                passed=bool(state.get("incident_resolved")),
                detail=(
                    "The agent declared the incident resolved after objective checks passed."
                    if state.get("incident_resolved")
                    else "The incident has not been safely declared resolved."
                ),
                weight=0.10,
            ),
            GraderCheck(
                name="speed_bonus_earned",
                passed=breakdown.get("speed_bonus", 0.0) > 0.0,
                detail=(
                    "Resolved faster than optimal_ticks."
                    if breakdown.get("speed_bonus", 0.0) > 0.0
                    else "Did not beat optimal tick budget."
                ),
                weight=0.10,
            ),
            GraderCheck(
                name="noise_handling",
                passed=breakdown.get("noise_handling_score", 0.0) >= 0.035,
                detail=(
                    "Minimal or no queries against noise services."
                    if breakdown.get("noise_handling_score", 0.0) >= 0.035
                    else "Wasted queries on noise services."
                ),
                weight=0.05,
            ),
        ]
        return GraderReport(
            scenario_id=scenario["id"],
            passed=passed,
            score=breakdown["final_score"],
            message=(
                "Incident diagnosed, remediated, and verified honestly."
                if passed
                else "Incident is not yet safely resolved."
            ),
            breakdown=breakdown,
            checks=report_checks,
        )
