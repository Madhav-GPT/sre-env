"""Advanced tier — horizon-bounded scenarios.

Three reference scenarios are shipped here as YAML specs.  They are *not*
runnable in this repo — the Advanced tier is documented as a roadmap (see
``docs/ADVANCED_TIER.md``) — but the YAML is real, structured, and can be
loaded via ``SREGym(tier=Tier.ADVANCED).list_scenarios()`` for inspection.

Why YAML and not Python dicts? Two reasons:

1. The Advanced tier has 15-20 service topologies, expanded action sets, and
   multi-incident chains. Encoding those as Python literals reads like noise.
2. Future scenario authors should never have to touch Python to add an
   Advanced scenario — see ``docs/SCENARIO_AUTHORING.md``.
"""
