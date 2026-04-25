"""sre-gym — tier-escalating SRE training environment.

Three tiers, three different escalation dimensions:

- ``Tier.BASIC``    — bounded by **compute**.  Pre-digested observations, 8K
  context, in-process simulator, 12 templates × 6 procgen variants = 72 scenarios.
  Trainable end-to-end on a single A100 in <12h. **The only tier shipped
  runnable; the other two are documented as design space.**
- ``Tier.ADVANCED`` — bounded by **horizon**.  Multi-incident sequences, 15-20
  service topology, partial-observability noise, ~25 actions.  Three concrete
  reference scenarios in ``sre_gym/advanced/scenarios/`` plus a design doc; not
  trainable in this repo.
- ``Tier.MAX``      — bounded by **realism**.  Ephemeral docker-compose / k3d
  stacks per ``reset()``, real ``kubectl rollout undo`` / Vercel API calls,
  Chaos-Mesh-style fault injection.  One fully-specced scenario family
  (e-commerce + Stripe + Supabase + Vercel) with ``compose.max.yaml`` and a
  Chaos-style spec; runnable infrastructure not provisioned in this repo.

The dimensional-escalation insight: each tier escalates a *different* axis, not
just scenario count.  See ``docs/ARCHITECTURE.md`` for the full rationale.

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
