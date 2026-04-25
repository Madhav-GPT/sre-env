"""Tier definitions for sre-gym.

Three tiers, three different bottlenecks, three different capabilities the
tier teaches the LLM:

- ``Tier.TRIAGE``     — causal mapping under tight compute
- ``Tier.STRATEGY``   — long-horizon planning across chained incidents
- ``Tier.OPERATIONS`` — authentic tool use against irreversible actions

Backwards compat: ``Tier.BASIC`` / ``Tier.ADVANCED`` / ``Tier.MAX`` still
resolve to the new members (Enum aliases). The module-level ``__getattr__``
emits a ``DeprecationWarning`` if external code does ``from sre_gym.tier
import BASIC`` so callers can migrate at their own pace; removal lands in v3.2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    """sre-gym tier.

    The primary names are TRIAGE / STRATEGY / OPERATIONS. The legacy names
    (BASIC / ADVANCED / MAX) are Enum aliases that resolve to the same
    members; ``Tier("basic")`` and ``Tier("triage")`` both return
    ``Tier.TRIAGE``.

    Persona mapping:
    - ``TRIAGE``     — student / Kaggle persona ($30 of HF credits, 1 A100 ~12h)
    - ``STRATEGY``   — startup / seed-stage persona ($300-500 budget, 1-2 A100 days)
    - ``OPERATIONS`` — enterprise persona (8x A100/H100, on-prem chaos engineering)
    """

    TRIAGE = "triage"
    STRATEGY = "strategy"
    OPERATIONS = "operations"

    # ---- Backwards-compat aliases (Enum collapses these onto the canonical members) ----
    BASIC = "triage"
    ADVANCED = "strategy"
    MAX = "operations"

    @classmethod
    def _missing_(cls, value: object) -> "Tier | None":
        """Accept legacy string values like ``"basic"``."""
        legacy_map = {
            "basic": cls.TRIAGE,
            "advanced": cls.STRATEGY,
            "max": cls.OPERATIONS,
        }
        if isinstance(value, str) and value.lower() in legacy_map:
            warnings.warn(
                f"Tier value {value!r} is deprecated; use "
                f"{legacy_map[value.lower()].value!r} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return legacy_map[value.lower()]
        return None


@dataclass(frozen=True)
class TierConfig:
    """Per-tier scaling/serving configuration.

    Each tier escalates a *different* dimension, not just scenario count. See
    ``docs/ARCHITECTURE.md`` for the full motivation and ``README.md`` for the
    one-paragraph form.
    """

    tier: Tier
    capability: str                    # what the tier teaches the LLM
    escalation_dimension: str          # "compute" | "horizon" | "realism"
    persona: str                       # one-line user persona
    expected_compute_budget: str       # human-readable
    scenario_count: int                # currently shipped runnable
    scenario_template_count: int
    procgen_variants_per_template: int
    expected_action_count: int
    max_episode_ticks: int
    observation_richness: str          # "pre-digested" | "noisy-multi-source" | "raw-real"
    runnable: bool
    runnable_kind: str                 # "live_environment" | "python_orchestrator" | "python_simulator"
    notes: str = ""
    docs_path: str = ""
    scenarios_glob: Optional[str] = None  # filepath glob for tier scenarios (data-only tiers)


TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.TRIAGE: TierConfig(
        tier=Tier.TRIAGE,
        capability="causal mapping under tight context",
        escalation_dimension="compute",
        persona="ML student / indie researcher with $30 of HF credits or 1 free Colab A100",
        expected_compute_budget="single A100 40GB for ~12h, or 1xL4 for ~25h",
        scenario_count=72,
        scenario_template_count=12,
        procgen_variants_per_template=5,
        expected_action_count=11,
        max_episode_ticks=13,
        observation_richness="pre-digested",
        runnable=True,
        runnable_kind="live_environment",
        notes=(
            "Topology is fixed at 4 services (api-gateway / cache / database / "
            "worker). Observations are pre-digested Four-Golden-Signals summaries "
            "so a full episode fits in 8K context. Reward shaping is dense: "
            "querying the right service before acting earns shaping credit."
        ),
        docs_path="docs/TRIAGE_TIER.md",
    ),
    Tier.STRATEGY: TierConfig(
        tier=Tier.STRATEGY,
        capability="long-horizon planning across chained incidents",
        escalation_dimension="horizon",
        persona="seed/Series A startup with $300-500 of compute or research lab with 1-2 A100 days",
        expected_compute_budget="1-2 A100 days for a 7B-14B LoRA + GRPO + DPO pass",
        scenario_count=3,
        scenario_template_count=3,
        procgen_variants_per_template=0,
        expected_action_count=28,
        max_episode_ticks=80,
        observation_richness="noisy-multi-source",
        runnable=True,
        runnable_kind="python_orchestrator",
        notes=(
            "Implemented as chained Triage episodes with persistent horizon state "
            "(unresolved alerts, pending deploys, tech-debt counter, horizon-decay "
            "reward). Each YAML declares a richer 28-action universe as design "
            "spec; the runner uses the Triage 11-action interface. Run via "
            "`python -m sre_gym.strategy run <scenario_id>`."
        ),
        docs_path="docs/STRATEGY_TIER.md",
        scenarios_glob="sre_gym/strategy/scenarios/*.yaml",
    ),
    Tier.OPERATIONS: TierConfig(
        tier=Tier.OPERATIONS,
        capability="authentic tool use against irreversible actions",
        escalation_dimension="realism",
        persona="enterprise SRE platform team with on-prem 8x A100/H100 cluster",
        expected_compute_budget="multi-day distributed training of a 32B-70B model",
        scenario_count=1,
        scenario_template_count=1,
        procgen_variants_per_template=0,
        expected_action_count=11,
        max_episode_ticks=40,
        observation_richness="raw-real",
        runnable=True,
        runnable_kind="python_simulator",
        notes=(
            "Python state-machine simulation over a 22-node service graph. Same "
            "11-action interface as Triage; chaos patterns are state-transition "
            "rules. The compose stack under sre_gym/operations/families/.../compose/ "
            "is design spec; ghcr.io/sre-gym/* images are not published. Run via "
            "`python -m sre_gym.operations run <family>` or "
            "`SREGym(tier=Tier.OPERATIONS).run(family_id, chaos=...)`."
        ),
        docs_path="docs/OPERATIONS_TIER.md",
        scenarios_glob="sre_gym/operations/families/*.yaml",
    ),
}


def describe_tier(tier: Tier) -> str:
    """Human-readable summary of a tier — used by the playground and CLI."""
    cfg = TIER_CONFIGS[tier]
    return (
        f"sre-gym tier={cfg.tier.value}\n"
        f"  capability           : {cfg.capability}\n"
        f"  escalation_dimension : {cfg.escalation_dimension}\n"
        f"  persona              : {cfg.persona}\n"
        f"  compute_budget       : {cfg.expected_compute_budget}\n"
        f"  scenarios            : {cfg.scenario_count} ({cfg.scenario_template_count} templates "
        f"× {cfg.procgen_variants_per_template + 1} entries)\n"
        f"  observation_richness : {cfg.observation_richness}\n"
        f"  runnable_kind        : {cfg.runnable_kind}\n"
        f"  docs                 : {cfg.docs_path}\n"
    )


# Module-level __getattr__ — emits a DeprecationWarning when external callers
# import a legacy alias by name (e.g. ``from sre_gym.tier import BASIC``).
# Attribute access on the Enum (``Tier.BASIC``) is silent because Enum aliases
# don't surface a hook for that path; the value-based ``Tier("basic")`` path
# warns via ``_missing_``.
_LEGACY_NAMES = {
    "BASIC": Tier.TRIAGE,
    "ADVANCED": Tier.STRATEGY,
    "MAX": Tier.OPERATIONS,
}


def __getattr__(name: str) -> Tier:  # pragma: no cover - exercised by deprecation tests
    if name in _LEGACY_NAMES:
        warnings.warn(
            f"sre_gym.tier.{name} is deprecated; use Tier.{_LEGACY_NAMES[name].name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_NAMES[name]
    raise AttributeError(f"module 'sre_gym.tier' has no attribute {name!r}")
