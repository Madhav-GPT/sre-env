"""Tier definitions for sre-gym.

The tier enum and config are deliberately data-only.  The escalation dimension
each tier targets is encoded in ``TierConfig.escalation_dimension`` and used by
docs/playground/training scripts to surface the design coherently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    """sre-gym difficulty tier.

    Mapping to operational personas:

    - ``BASIC``    — student / Kaggle persona ($30 of HF credits, 1 A100 @ 12h)
    - ``ADVANCED`` — startup / seed-stage persona ($300-500 budget, 1-2 A100 days)
    - ``MAX``      — enterprise persona (8x A100/H100, on-prem chaos engineering)
    """

    BASIC = "basic"
    ADVANCED = "advanced"
    MAX = "max"


@dataclass(frozen=True)
class TierConfig:
    """Per-tier scaling/serving configuration.

    Each tier escalates a *different* dimension, not just scenario count.  This
    is the single insight that ties the whole pitch together — see the README's
    first paragraph and ``docs/ARCHITECTURE.md`` for the full motivation.
    """

    tier: Tier
    escalation_dimension: str          # "compute" | "horizon" | "realism"
    persona: str                       # one-line user persona
    expected_compute_budget: str       # human-readable
    scenario_count: int                # currently shipped runnable
    scenario_template_count: int
    procgen_variants_per_template: int
    expected_action_count: int
    max_episode_ticks: int
    observation_richness: str          # "pre-digested" | "noisy-multi-source" | "raw-real"
    runnable: bool                     # is this tier executable in this repo?
    notes: str = ""
    docs_path: str = ""
    scenarios_glob: Optional[str] = None  # filepath glob for tier scenarios (data-only tiers)


TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.BASIC: TierConfig(
        tier=Tier.BASIC,
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
        notes=(
            "Scenarios are causally rich (8-service topology, full deploy history, "
            "evidence trail) but observations are pre-digested (Four Golden Signals, "
            "error-signature summaries, deploy diffs) so a full episode fits in 8K "
            "context. Reward shaping is dense; querying the right service before "
            "acting earns shaping reward."
        ),
        docs_path="docs/BASIC_TIER.md",
    ),
    Tier.ADVANCED: TierConfig(
        tier=Tier.ADVANCED,
        escalation_dimension="horizon",
        persona="seed/Series A startup with $300-500 of compute or research lab with 1-2 A100 days",
        expected_compute_budget="1-2 A100 days for a 7B-14B LoRA + GRPO + DPO pass",
        scenario_count=3,                # 3 reference scenarios shipped
        scenario_template_count=3,
        procgen_variants_per_template=0,
        expected_action_count=28,
        max_episode_ticks=80,
        observation_richness="noisy-multi-source",
        runnable=True,                   # chained Basic episodes via sre_gym.advanced.runner
        notes=(
            "Wider 15-20 service topology, multi-incident sequences (one fix introduces "
            "a downstream incident), partial-observability noise (the logging pipeline "
            "itself can be the affected service), expanded action set with traces / PRs "
            "/ feature-flag toggles / on-call escalation. Implemented as chained Basic "
            "episodes with persistent horizon state (unresolved alerts, pending deploys, "
            "tech-debt counter, horizon-decay reward). Run via "
            "`python -m sre_gym.advanced run <scenario_id>` or "
            "`SREGym(tier=Tier.ADVANCED).run(scenario_id)`."
        ),
        docs_path="docs/ADVANCED_TIER.md",
        scenarios_glob="sre_gym/advanced/scenarios/*.yaml",
    ),
    Tier.MAX: TierConfig(
        tier=Tier.MAX,
        escalation_dimension="realism",
        persona="enterprise SRE platform team with on-prem 8x A100/H100 cluster",
        expected_compute_budget="multi-day distributed training of a 32B-70B model",
        scenario_count=1,                # one fully-specced family shipped
        scenario_template_count=1,
        procgen_variants_per_template=0,
        expected_action_count=11,        # Python simulator reuses Basic 11-action interface
        max_episode_ticks=40,
        observation_richness="raw-real",
        runnable=True,                   # Python state-machine simulation via sre_gym.max.runner
        notes=(
            "The Space runs Max as a Python state-machine simulation over the "
            "22-node service graph; actions mutate graph state and reward reuses the "
            "Basic potential-shaped function over graph health. The real docker-compose "
            "stack lives in sre_gym/max/families/ecommerce_vibecoded_saas/compose/ and "
            "can be run locally with `docker compose up` for users with Docker. The "
            "Python simulator is faithful to the chaos patterns but does not reproduce "
            "real-network failure modes. Run via `python -m sre_gym.max run <family>` "
            "or `SREGym(tier=Tier.MAX).run(family_id, chaos=...)`."
        ),
        docs_path="docs/MAX_TIER.md",
        scenarios_glob="sre_gym/max/families/*.yaml",
    ),
}


def describe_tier(tier: Tier) -> str:
    """Human-readable summary of a tier — used by the playground and CLI."""
    cfg = TIER_CONFIGS[tier]
    return (
        f"sre-gym tier={cfg.tier.value}\n"
        f"  escalation_dimension : {cfg.escalation_dimension}\n"
        f"  persona              : {cfg.persona}\n"
        f"  compute_budget       : {cfg.expected_compute_budget}\n"
        f"  scenarios            : {cfg.scenario_count} ({cfg.scenario_template_count} templates "
        f"× {cfg.procgen_variants_per_template + 1} variants)\n"
        f"  observation_richness : {cfg.observation_richness}\n"
        f"  runnable_in_repo     : {cfg.runnable}\n"
        f"  docs                 : {cfg.docs_path}\n"
    )
