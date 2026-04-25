"""sre-gym — tier-escalating SRE training environment.

Three tiers, three different escalation dimensions:

- ``Tier.BASIC``    — bounded by **compute**.  Pre-digested observations, 8K
  context, in-process simulator, 12 base templates each with 5 procgen
  variants (so 6 entries per template, 72 total scenarios).  The only tier
  whose runner is a live HTTP environment.

- ``Tier.ADVANCED`` — bounded by **horizon**.  A Python orchestrator that
  chains Basic episodes together with persistent horizon state (unresolved
  alerts, pending deploys, tech-debt counter, horizon-decay reward).
  ``sre_gym/advanced/scenarios/*.yaml`` declare a richer ~28-action universe
  as **design spec**; those extra actions are NOT implemented in the env.
  The runner falls back to the Basic 11-action interface.

- ``Tier.MAX``      — bounded by **realism**.  A Python state-machine
  simulator over a 22-node service graph.  Same 11 Basic actions; chaos
  patterns are state-transition rules over the in-memory graph.
  ``sre_gym/max/families/.../compose/ecommerce.yaml`` describes a real
  docker-compose stack but the stub images (``ghcr.io/sre-gym/...``) are
  NOT published; treat the compose file as design-spec only.

All three tiers are *runnable* via ``SREGym(tier=...).run(...)`` and the
respective ``python -m sre_gym.<tier> run ...`` CLIs.  Only Basic exposes a
live HTTP environment with /reset + /step routes; Advanced and Max are
in-process orchestrators.

Training (notebooks/01_basic_train_grpo_unsloth.ipynb) is **not committed to
this repo**.  Run it externally on Colab A100 and publish the resulting
adapter + comparison plots to ``eval/results/``.

The dimensional-escalation insight: each tier escalates a *different* axis,
not just scenario count.  See ``docs/ARCHITECTURE.md`` for the full rationale.

Quick start
-----------
::

    from sre_gym import SREGym, Tier

    env = SREGym(tier=Tier.BASIC)
    obs = env.reset(scenario_id="memory_leak_oom__p02")
    obs = env.step({"action_type": "rollback_deploy", "service": "worker"})

The Basic tier delegates to ``unified_incident_env``, which is shipped as the
HF Space's runnable surface and is what ``openenv.yaml`` declares.
"""

from __future__ import annotations

from .env import SREGym
from .tier import Tier, TierConfig, TIER_CONFIGS

__all__ = ["SREGym", "Tier", "TierConfig", "TIER_CONFIGS", "__version__"]
__version__ = "1.0.0"
