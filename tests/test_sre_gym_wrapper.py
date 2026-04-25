"""Tests for the sre_gym tier-aware wrapper.

Three things to lock down:

1. Basic tier delegates to UnifiedIncidentEnvironment cleanly.
2. Advanced and Max tiers expose their YAML spec via list_scenarios but raise
   TierNotRunnableError on reset/step.
3. The Tier enum + TierConfig escalation_dimension story is consistent with
   docs/ARCHITECTURE.md (the dimensional-escalation insight).
"""

from __future__ import annotations

import pytest

from sre_gym import SREGym, Tier, TIER_CONFIGS
from sre_gym.env import TierNotRunnableError


# ---------- Basic tier ----------

def test_basic_tier_runs_end_to_end() -> None:
    env = SREGym(tier=Tier.BASIC)
    obs = env.reset(scenario_id="worker_deploy_cascade")
    assert obs.workflow_stage == "triage"
    assert len(obs.active_alerts) > 0


def test_basic_tier_steps_correctly() -> None:
    env = SREGym(tier=Tier.BASIC)
    env.reset(scenario_id="worker_deploy_cascade")
    obs = env.step({"action_type": "query_deploys", "service": "worker"})
    assert obs.tool_output is not None
    assert "worker" in obs.tool_output.lower()


def test_basic_tier_lists_72_scenarios() -> None:
    env = SREGym(tier=Tier.BASIC)
    scenarios = env.list_scenarios()
    assert len(scenarios) == 72


# ---------- Advanced tier ----------

def test_advanced_tier_lists_three_reference_scenarios() -> None:
    env = SREGym(tier=Tier.ADVANCED)
    scenarios = env.list_scenarios()
    ids = {s["id"] for s in scenarios}
    assert {
        "cascading_release_train",
        "observability_pipeline_outage",
        "supabase_rls_silent_leak",
    } == ids


def test_advanced_tier_reset_raises_with_docs_pointer() -> None:
    env = SREGym(tier=Tier.ADVANCED)
    with pytest.raises(TierNotRunnableError) as info:
        env.reset()
    assert "docs/ADVANCED_TIER.md" in info.value.docs_path


def test_advanced_scenarios_have_horizon_dimension_signals() -> None:
    """Each Advanced scenario must demonstrate the horizon escalation."""
    env = SREGym(tier=Tier.ADVANCED)
    for spec in env.list_scenarios():
        assert spec.get("max_ticks", 0) >= 60, (
            f"{spec['id']}: max_ticks={spec.get('max_ticks')} (must be >=60 for horizon tier)"
        )
        # Either has multi-phase incident_chain or a long reference_trace
        chain = spec.get("incident_chain", [])
        trace = spec.get("reference_trace", {})
        assert len(chain) >= 2 or len(trace) >= 2, (
            f"{spec['id']}: must have multi-phase incident_chain or trace"
        )


# ---------- Max tier ----------

def test_max_tier_lists_one_family() -> None:
    env = SREGym(tier=Tier.MAX)
    scenarios = env.list_scenarios()
    assert len(scenarios) == 1
    fam = scenarios[0]
    assert fam["id"] == "ecommerce_vibecoded_saas"
    assert len(fam["topology"]["services"]) >= 20
    assert fam["scenario_population"]["size"] >= 30


def test_max_tier_reset_returns_observation() -> None:
    """Max tier is now runnable via the Python state-machine simulator."""
    env = SREGym(tier=Tier.MAX)
    obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="deploy_regression", seed=1)
    assert obs.family_id == "ecommerce_vibecoded_saas"
    assert obs.chaos == "deploy_regression"
    assert len(obs.services) >= 20


def test_advanced_tier_run_method_returns_result() -> None:
    """Advanced tier is now runnable via the chained-Basic-episodes runner."""
    env = SREGym(tier=Tier.ADVANCED)
    result = env.run("cascading_release_train", seed=1)
    assert result.scenario_id == "cascading_release_train"
    assert len(result.phases) == 2
    assert result.success is True
    assert 0.5 < result.final_reward < 0.85


def test_advanced_tier_reset_still_raises_for_per_step_caller() -> None:
    """Advanced is episodic — direct reset() must still raise."""
    env = SREGym(tier=Tier.ADVANCED)
    with pytest.raises(TierNotRunnableError) as info:
        env.reset()
    assert "docs/ADVANCED_TIER.md" in info.value.docs_path
    assert "run(scenario_id)" in str(info.value)


def test_max_family_has_operator_notes() -> None:
    env = SREGym(tier=Tier.MAX)
    fam = env.list_scenarios()[0]
    notes = fam["operator_notes"]
    assert "cost_estimate" in notes
    assert "isolation_requirements" in notes
    assert "reset_safety" in notes


# ---------- Tier-config consistency (the dimensional-escalation defence) ----------

def test_tier_configs_have_distinct_escalation_dimensions() -> None:
    dims = [TIER_CONFIGS[t].escalation_dimension for t in Tier]
    assert dims == ["compute", "horizon", "realism"], (
        f"Expected the dimensional-escalation insight, got {dims}. "
        f"See docs/ARCHITECTURE.md."
    )


def test_describe_includes_escalation_dimension() -> None:
    for tier in Tier:
        env = SREGym(tier=tier)
        info = env.describe()
        assert "escalation_dimension" in info
        assert info["escalation_dimension"] in ("compute", "horizon", "realism")


def test_describe_includes_persona_and_compute_budget() -> None:
    for tier in Tier:
        env = SREGym(tier=tier)
        info = env.describe()
        assert info["persona"]
        assert info["compute_budget"]
        assert info["docs"]
