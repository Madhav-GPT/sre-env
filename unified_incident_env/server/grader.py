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
    """Deterministic scorer focused on executed effects, not scripted clues."""

    def compute_breakdown(
        self,
        state: dict[str, Any],
        scenario: dict[str, Any],
    ) -> dict[str, float]:
        services = state.get("service_health", {})
        weights = scenario["critical_service_weights"]
        recovery_score = round(
            sum(
                weights.get(service, 0.0) * _service_score((services.get(service) or {}).get("status", "crashed"))
                for service in weights
            ),
            4,
        )

        containment_score = 0.2 if state.get("containment_applied") else 0.0
        if state.get("containment_applied") and (services.get("worker") or {}).get("status") == "healthy":
            containment_score = 0.3

        checks = {item.get("name"): bool(item.get("passed")) for item in state.get("checks", [])}
        verification_score = 0.0
        if checks.get("database_recovery"):
            verification_score += 0.15
        if checks.get("end_to_end"):
            verification_score += 0.2

        user_impact = float(state.get("user_impact", 1.0))
        impact_score = round(max(0.0, 0.15 * (1.0 - user_impact)), 4)

        wasteful_ticks = int(state.get("wasteful_ticks", 0))
        efficiency_score = round(max(0.0, 0.10 - (0.01 * wasteful_ticks)), 4)

        final_score = _strict_public_score(
            recovery_score + containment_score + verification_score + impact_score + efficiency_score
        )

        return {
            "recovery_score": recovery_score,
            "containment_score": round(containment_score, 4),
            "verification_score": round(verification_score, 4),
            "impact_score": impact_score,
            "efficiency_score": efficiency_score,
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
                weight=0.30,
            ),
            GraderCheck(
                name="database_recovery",
                passed=checks.get("database_recovery", False),
                detail=(
                    "The database recovery check passed."
                    if checks.get("database_recovery")
                    else "The database recovery check has not passed yet."
                ),
                weight=0.20,
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
                passed=breakdown["recovery_score"] >= 0.8,
                detail=(
                    "Critical-path services are recovered."
                    if breakdown["recovery_score"] >= 0.8
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
