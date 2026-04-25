"""Smoke tests for the 6 round-2 Basic-tier templates added for the OpenEnv hackathon.

Each template must:
1. Resolve cleanly via its scripted-optimal baseline.
2. Score in the 0.70-0.80 band (not below — that means the baseline is bad;
   not above — that means the rubric is leaking and the trained-agent
   headroom is gone).
3. Carry at least one noise service so the noise_handling_score dimension is alive.
4. Have all five procgen variants resolvable via the same baseline structure.
5. Reject wrong-target rollbacks with the documented failure_type.
"""

from __future__ import annotations

import pytest

from unified_incident_env.server.challenge import SCENARIOS, get_scenario, list_baselines
from unified_incident_env.server.environment import UnifiedIncidentEnvironment


ROUND2_TEMPLATES = [
    "dep_degradation",
    "memory_leak_oom",
    "auth_token_expiry",
    "network_partition",
    "rate_limit_retry_storm",
    "migration_lock",
]


def _walk_baseline(scenario_id: str):
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=scenario_id)
    baseline = list_baselines(scenario_id=scenario_id).baselines[0]
    last = None
    for step in baseline.actions:
        last = env.step(step.action)
        if last.done:
            break
    return last


@pytest.mark.parametrize("template_id", ROUND2_TEMPLATES)
def test_round2_baseline_resolves(template_id: str) -> None:
    obs = _walk_baseline(template_id)
    assert obs is not None
    assert obs.incident_resolved is True, (
        f"{template_id}: baseline failed to resolve (failure_type={obs.failure_type}, "
        f"why={obs.why_failed})"
    )
    assert 0.70 <= obs.final_score <= 0.80, (
        f"{template_id}: baseline scored {obs.final_score:.3f} "
        f"(must be in [0.70, 0.80] — see docs/REWARD_DESIGN.md)"
    )


@pytest.mark.parametrize("template_id", ROUND2_TEMPLATES)
def test_round2_template_has_noise_services(template_id: str) -> None:
    spec = get_scenario(template_id)
    knobs = spec.get("difficulty_knobs", {})
    assert knobs.get("noise_services"), f"{template_id}: must declare at least one noise service"
    assert knobs.get("noise_alerts"), f"{template_id}: noise_services must come with noise_alerts"
    assert knobs.get("noise_logs"), f"{template_id}: noise_services must come with noise_logs"


@pytest.mark.parametrize("template_id", ROUND2_TEMPLATES)
def test_round2_procgen_variants_resolve(template_id: str) -> None:
    """All 5 procgen variants must resolve via the same baseline structure."""
    variant_ids = [f"{template_id}__p{i:02d}" for i in range(1, 6)]
    for vid in variant_ids:
        assert vid in SCENARIOS, f"missing procgen variant {vid}"
        obs = _walk_baseline(vid)
        assert obs.incident_resolved is True, (
            f"{vid}: procgen variant failed to resolve"
        )
        assert obs.final_score >= 0.70, (
            f"{vid}: procgen variant scored {obs.final_score:.3f} (must be >= 0.70)"
        )


@pytest.mark.parametrize("template_id", ROUND2_TEMPLATES)
def test_round2_wrong_rollback_target_penalized(template_id: str) -> None:
    """A wrong-target rollback should set failure_type='wrong_remediation_target'."""
    spec = get_scenario(template_id)
    correct_target = spec["remediation_recipe"].get("rollback_target")
    candidates = ["api-gateway", "cache", "database", "worker"]
    wrong_targets = [s for s in candidates if s != correct_target]
    assert wrong_targets, f"{template_id}: needed at least one wrong-target candidate"
    wrong = wrong_targets[0]

    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=template_id)
    from unified_incident_env.models import UnifiedIncidentAction
    obs = env.step(UnifiedIncidentAction(action_type="rollback_deploy", service=wrong))
    assert obs.failure_type == "wrong_remediation_target", (
        f"{template_id}: expected wrong_remediation_target failure on rollback({wrong}), "
        f"got failure_type={obs.failure_type}"
    )
    assert obs.reward < 0.0, (
        f"{template_id}: wrong-target rollback should be penalized (got reward={obs.reward})"
    )


def test_total_template_count() -> None:
    """Catalogue should contain 12 templates and 72 total scenarios (12 + 60 procgen)."""
    template_ids = sorted({s.get("template_id") for s in SCENARIOS.values()})
    assert len(template_ids) == 12, (
        f"Expected 12 templates after round-2 additions, got {len(template_ids)}: {template_ids}"
    )
    total_scenarios = len(SCENARIOS)
    expected = 12 * 6  # 12 base + 5 procgen each
    assert total_scenarios == expected, (
        f"Expected {expected} scenarios (12 base + 60 procgen), got {total_scenarios}"
    )


def test_round2_template_ids_present() -> None:
    """All 6 round-2 template IDs must show up in the catalogue."""
    template_ids = {s.get("template_id") for s in SCENARIOS.values()}
    for tid in ROUND2_TEMPLATES:
        assert tid in template_ids, f"{tid} not in template catalogue"
