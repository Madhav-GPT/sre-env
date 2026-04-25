"""Max tier — realism-bounded scenario families.

The Max tier is the apex tier of sre-gym: an ephemeral docker-compose / k3d
sandbox is provisioned per ``reset()``, the agent's actions are real
``kubectl rollout undo`` / Vercel API calls / Supabase admin actions, faults
are injected via real Chaos-Mesh-style patterns, and the reward is computed
from the actual recovery state of the actual stack.

This package ships ONE fully-specced scenario family
(``ecommerce_vibecoded_saas``) with:

- ``compose/ecommerce.yaml``                — the docker-compose stack
- ``chaos/ecommerce_chaos_library.yaml``    — the chaos-injection catalogue
- ``families/ecommerce_vibecoded_saas.yaml`` — the family-level scenario spec
- ``docs/MAX_TIER.md``                       — architecture and roadmap

The infrastructure is not provisioned in this repo — bringing the Max tier up
costs roughly $40-150/day depending on cluster size, and is the appropriate
investment for a Series-A SRE platform team or a research lab fine-tuning a
32B-70B model. The artifacts here are real and structured so that downstream
operator can lift them into a production-grade RL training cluster.
"""
