"""Smoke tests for session/report shells after the v2 pivot."""

from __future__ import annotations

from unified_incident_env.trainer.reporting import build_phase_deltas
from unified_incident_env.trainer.types import SessionPhaseReport


def test_build_phase_deltas_handles_simple_progression() -> None:
    phases = [
        SessionPhaseReport(
            phase_name="probe",
            episode_ids=[1, 2],
            avg_score=0.2,
            success_rate=0.0,
            schema_failures=1,
            loop_failures=1,
            updates_applied=[],
        ),
        SessionPhaseReport(
            phase_name="final_evaluation",
            episode_ids=[3, 4],
            avg_score=0.8,
            success_rate=1.0,
            schema_failures=0,
            loop_failures=0,
            updates_applied=[],
        ),
    ]
    deltas = build_phase_deltas(phases)
    assert deltas[1].phase_name == "final_evaluation"
    assert deltas[1].score_delta == 0.6
