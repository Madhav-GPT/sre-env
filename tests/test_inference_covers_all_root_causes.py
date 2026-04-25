"""Asserts inference.py's hypothesis schema covers every RootCauseType.

If a contributor adds a new root cause to ``unified_incident_env.models`` but
forgets to extend ``inference._ROOT_CAUSE_ENUM``, the model can no longer
emit a hypothesis with that root cause through the JSON-schema-constrained
sampling path. This test fails CI in that case.
"""

from __future__ import annotations

from unified_incident_env.models import RootCauseType


def test_inference_root_cause_enum_covers_models_literal() -> None:
    """Every RootCauseType in models.py must appear in inference._ROOT_CAUSE_ENUM."""
    import inference  # repo-root inference.py

    canonical = set(RootCauseType.__args__)            # type: ignore[attr-defined]
    declared = set(inference._ROOT_CAUSE_ENUM)
    missing = canonical - declared
    extra = declared - canonical
    assert not missing, (
        f"inference._ROOT_CAUSE_ENUM is missing {missing}. "
        f"Add them to inference.py (see basic_templates_extra.py for round-2 IDs)."
    )
    assert not extra, (
        f"inference._ROOT_CAUSE_ENUM has unknown values {extra}. "
        f"Either add them to RootCauseType in models.py or remove them here."
    )


def test_inference_root_cause_enum_has_twelve_entries() -> None:
    """The catalogue ships 12 root causes; the enum must mirror that count."""
    import inference

    assert len(inference._ROOT_CAUSE_ENUM) == 12, (
        f"Expected 12 root causes (6 v2 + 6 round-2), got {len(inference._ROOT_CAUSE_ENUM)}"
    )


def test_heuristic_fallback_emits_valid_root_cause_for_each_template() -> None:
    """For every Basic template, the heuristic fallback emits a root cause that
    parses against ``UnifiedIncidentAction.hypothesis``.

    This is the resilience guarantee: when the LLM is offline, the fallback
    produces a structurally valid hypothesis that scores at least partial credit.
    """
    import inference

    from unified_incident_env.server.environment import UnifiedIncidentEnvironment
    from unified_incident_env.server.challenge import SCENARIOS
    from unified_incident_env.models import UnifiedIncidentAction

    template_ids = sorted({s.get("template_id") for s in SCENARIOS.values()})
    assert len(template_ids) == 12

    for tid in template_ids:
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=tid)
        root_cause, affected = inference._heuristic_root_cause(obs)
        # The action must construct without raising — otherwise the heuristic
        # is emitting a malformed hypothesis.
        action = UnifiedIncidentAction(
            action_type="submit_hypothesis",
            hypothesis={
                "root_cause": root_cause,
                "affected_services": affected,
                "confidence": 0.6,
                "recommended_next_action": "rollback_deploy",
            },
        )
        assert action.hypothesis is not None
        assert action.hypothesis.root_cause == root_cause
