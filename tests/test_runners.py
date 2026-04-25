"""End-to-end tests for the Advanced and Max runners.

The runners live in ``sre_gym.advanced.runner`` and ``sre_gym.max.runner``.
These tests assert that:

- All 3 Advanced reference scenarios can be run with the default
  scripted-optimal policy and produce a coherent ``AdvancedResult``.
- Every Max chaos pattern is reachable by ``run_max(...)`` against the
  ``ecommerce_vibecoded_saas`` family without raising.
- The runners produce deterministic outputs given the same seed.
"""

from __future__ import annotations


import pytest

from sre_gym.advanced.runner import (
    AdvancedResult,
    list_advanced_scenarios,
    run_advanced,
)
from sre_gym.max.runner import (
    CHAOS_PATTERNS,
    MaxResult,
    list_max_families,
    run_max,
)


# ---------- Advanced runner ----------


def test_list_advanced_scenarios_has_three_entries() -> None:
    scenarios = list_advanced_scenarios()
    assert {"cascading_release_train", "observability_pipeline_outage", "supabase_rls_silent_leak"}.issubset(
        set(scenarios)
    )


@pytest.mark.parametrize(
    "scenario_id",
    ["cascading_release_train", "observability_pipeline_outage", "supabase_rls_silent_leak"],
)
def test_advanced_scenario_runs_end_to_end(scenario_id: str) -> None:
    result = run_advanced(scenario_id, seed=1)
    assert isinstance(result, AdvancedResult)
    assert result.scenario_id == scenario_id
    assert len(result.phases) >= 1
    # Every phase must produce a final score in [0.0, 1.0].
    for phase in result.phases:
        assert 0.0 <= phase.final_score <= 1.0
    # Horizon-decay ≤ 1, raw mean ≥ 0.
    assert 0.0 <= result.horizon_decay_factor <= 1.0
    assert 0.0 <= result.raw_mean_reward <= 1.0


def test_advanced_runner_is_deterministic_for_same_seed() -> None:
    a = run_advanced("cascading_release_train", seed=42)
    b = run_advanced("cascading_release_train", seed=42)
    assert a.final_reward == b.final_reward
    assert a.raw_mean_reward == b.raw_mean_reward
    assert len(a.phases) == len(b.phases)


def test_advanced_emits_log_lines_when_callback_provided() -> None:
    captured: list[str] = []
    run_advanced("cascading_release_train", seed=1, on_log=captured.append)
    assert any("phase 1" in line for line in captured)
    assert any("phase 2" in line for line in captured)
    assert any("declare_resolved" in line for line in captured)


# ---------- Max runner ----------


def test_list_max_families_has_ecommerce_family() -> None:
    families = list_max_families()
    assert "ecommerce_vibecoded_saas" in families


@pytest.mark.parametrize("chaos", list(CHAOS_PATTERNS))
def test_max_chaos_pattern_runs_without_error(chaos: str) -> None:
    """Every documented chaos pattern must be reachable from run_max."""
    result = run_max("ecommerce_vibecoded_saas", chaos=chaos, seed=1)
    assert isinstance(result, MaxResult)
    assert result.chaos == chaos
    assert 0.0 <= result.final_reward <= 1.0
    assert 0 <= result.tick_count <= 25


def test_max_runner_is_deterministic_for_same_seed() -> None:
    a = run_max("ecommerce_vibecoded_saas", chaos="deploy_regression", seed=42)
    b = run_max("ecommerce_vibecoded_saas", chaos="deploy_regression", seed=42)
    assert a.final_reward == b.final_reward
    assert a.cumulative_reward == b.cumulative_reward
    assert a.tick_count == b.tick_count


def test_max_security_classified_chaos_carries_classification() -> None:
    """rls_silent_leak / oauth_supply_chain_pivot / cdn_cache_contamination
    must surface ``classification == 'security'`` in MaxResult so the UI
    can render a security badge."""
    for chaos in ("rls_silent_leak", "oauth_supply_chain_pivot", "cdn_cache_contamination"):
        result = run_max("ecommerce_vibecoded_saas", chaos=chaos, seed=1)
        assert result.classification == "security", (
            f"{chaos} must be classification='security' (got {result.classification!r})"
        )


def test_max_per_step_env_returns_observation_on_reset() -> None:
    """The MaxRunnerEnv (used by SREGym(tier=Tier.MAX).reset/step) returns a
    valid observation."""
    from sre_gym.max.runner import MaxRunnerEnv

    env = MaxRunnerEnv(family_id="ecommerce_vibecoded_saas")
    obs = env.reset(chaos="deploy_regression", seed=1)
    assert obs.tick_count == 0
    assert obs.family_id == "ecommerce_vibecoded_saas"
    assert obs.chaos == "deploy_regression"
    assert "api-gateway" in obs.services


def test_max_per_step_env_steps() -> None:
    from sre_gym.max.runner import MaxRunnerEnv

    env = MaxRunnerEnv()
    env.reset(chaos="deploy_regression", seed=1)
    obs = env.step({"action_type": "query_logs", "service": "orders-service"})
    assert obs.tick_count == 1
    assert "query" in obs.last_log
