"""Smoke tests for reusable trainer-shell pieces after the v2 pivot."""

from __future__ import annotations

from pathlib import Path

from unified_incident_env.trainer.trajectory_memory import CorrectionMemory
from unified_incident_env.trainer.trajectory_store import TrajectoryStore
from unified_incident_env.trainer.types import EpisodeRecord, StepRecord


def test_correction_memory_empty_prompt_is_safe() -> None:
    memory = CorrectionMemory()
    addendum = memory.build_prompt_addendum("worker_deploy_cascade", "triage")
    assert isinstance(addendum, str)


def test_trajectory_store_roundtrip(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "episodes.jsonl")
    record = EpisodeRecord(
        run_id="run-1",
        scenario_id="worker_deploy_cascade",
        difficulty="easy",
        model_name="stub",
        mode="strict",
        success=False,
        final_score=0.1,
        steps=1,
        elapsed_s=0.01,
        step_records=[
            StepRecord(
                step_index=1,
                tick=1,
                workflow_stage="triage",
                observation={},
                prompt_text="prompt",
                raw_model_output="{}",
                parse_status="invalid_json",
                reward=None,
            )
        ],
    )
    store.append_episode(record)
    loaded = store.load_episodes()
    assert len(loaded) == 1
    assert loaded[0].scenario_id == "worker_deploy_cascade"
