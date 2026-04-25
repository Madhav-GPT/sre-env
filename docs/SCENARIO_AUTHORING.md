# Scenario authoring guide

> How to add a 13th template (Basic), a fourth reference scenario (Advanced), or a second family (Max). The rule is: **scenarios are pure data; the simulator core never needs to change.**

---

## 1. Adding a Basic-tier template (the 60-minute path)

A new Basic template is a Python dict added to either `_BASE_SCENARIOS` (in `unified_incident_env/server/challenge.py`) or `EXTRA_TEMPLATES` (in `unified_incident_env/server/basic_templates_extra.py`). Recommended: use the latter, since it's where the round-2 templates live and it keeps `challenge.py` readable.

### Required keys

```python
{
    "id":                       "your_template_id",
    "difficulty":               "easy" | "medium" | "hard",
    "name":                     "Human-readable title",
    "description":              "1-3 sentence summary of the incident shape",
    "root_cause":               "1 sentence root cause description",
    "optimal_ticks":            8 | 9 | 10 | 11 | 12,
    "max_ticks":                10 | 12 | 13,
    "critical_service_weights": {"api-gateway": ..., "cache": ..., "database": ..., "worker": ...},  # sums to 1.0
    "reward_config":            <reward dict>,                       # see basic_templates_extra._STD_REWARD
    "initial_services":         {<service>: {<status, cpu, mem, err, latency>}},
    "initial_alerts":           [<alert>, ...],
    "logs":                     {<service>: <single-line digest>},
    "metrics":                  {<service>: {<metric>: <single-line digest>}},
    "dependencies":             {<service>: <single-line description>},
    "deploy_history":           {<service>: <single-line history>},
    "checks":                   {"database_recovery": <description>, "end_to_end": <description>},
    "truth":                    {"root_cause": <RootCauseType>, "affected_services": [...], "best_next_action": <RecommendedActionType>},
    "remediation_recipe":       {<rollback_target, restart_target, isolate_target, restart_requires_cause_removed, incident_driver, resolution_check>},
    "post_rollback_services":   {<service>: <new health snapshot>},
    "post_restart_services":    {<service>: <new health snapshot>},
    "post_isolate_services":    {<service>: <new health snapshot>},
    "post_rollback_user_impact": <float in 0..1>,
    "post_rollback_slo_burn":    <float in 0..1>,
    "post_restart_user_impact":  <float in 0..1>,
    "post_restart_slo_burn":     <float in 0..1>,
    "post_isolate_user_impact":  <float in 0..1>,
    "post_isolate_slo_burn":     <float in 0..1>,
    "degraded_services":        {<service>: <baseline degraded snapshot>},
    "degraded_user_impact":     <float>,
    "degraded_slo_burn":        <float>,
    "failure_messages":         {<failure_type>: <message>},
    "difficulty_knobs":         {"noise_services": [...], "noise_alerts": [...], "noise_logs": {...}, "blast_radius_budget": <int>},
}
```

### Required follow-ups

1. **Add a baseline.** Append a lambda to the dict returned by `extra_baselines()` in `basic_templates_extra.py`:
   ```python
   "your_template_id": lambda: [
       _ba("query_logs", service="...", rationale="..."),
       ...
       _ba("declare_resolved", rationale="..."),
   ],
   ```
   The baseline must resolve the scenario in `optimal_ticks` steps and score in the 0.70–0.80 band.

2. **Add the root_cause to the Literal.** In `unified_incident_env/models.py`, append your new `RootCauseType` value. This is what `submit_hypothesis` validates against.

3. **Add a smoke test.** In `unified_incident_env/tests/`, add a test that walks the baseline and asserts `obs.incident_resolved is True` and `obs.final_score >= 0.70`.

That's it. Procedural generation picks up the new template automatically (5 procgen variants per template), `list_scenarios()` exposes them at `/tasks`, and the grader scores them with the same 7-dimension rubric.

### Anti-patterns

- **Don't tune the rubric for your new template.** If your template scores below 0.70, the *scenario* is wrong (probably `optimal_ticks` is too generous or post-rollback states aren't healthy enough). Don't bump rubric weights.
- **Don't skip noise services.** Every scenario needs at least 1–2 distractor noise services. Without them, `noise_handling_score` is dead-weight reward.
- **Don't make the deploy_history single-service.** Every scenario should have *at least one* decoy deploy on a non-culprit service. Otherwise the agent learns to grep deploy timestamps for the most-recent and short-circuit reasoning.

---

## 2. Adding an Advanced-tier reference scenario (the 90-minute path)

Drop a new YAML file in `sre_gym/advanced/scenarios/` matching the schema in [`docs/ADVANCED_TIER.md`](ADVANCED_TIER.md) §2. Required sections:

- `id` / `tier: advanced` / `difficulty` / `name` / `description`
- `topology` — 15–20 services with `id`, `kind`, `owner`
- `incident_chain` — multi-phase incident definition with `triggered_by`, `failing_services`, `correct_action`, `deceptive_signal` per phase
- `allowed_actions` — explicit list (inherits Basic by convention but you can override)
- `reward_dimensions` — must include the Basic 7 plus any Advanced-tier additions
- `reference_trajectory_length` / `optimal_ticks` / `max_ticks`
- `reference_trace` — the canonical optimal path, formatted as `phase_N: [{tick, action, expected_signal}, ...]`
- `oncall_peer` — synthetic peer behaviour spec (optional)
- `success_criteria` — boolean checks that determine if the agent passed

The reference trace is *the* most important section: it's the SFT seed-data shape and the documentation of what "good" looks like for the scenario. Skip it and the spec is unactionable.

### Why YAML, not Python

- Advanced scenarios have rich nested structure (multi-phase incidents, oncall_peer behaviours, success_criteria) that reads as noise in Python literals.
- Future scenario authors should never have to touch Python to add an Advanced scenario — same as Litmus chaos experiments.

---

## 3. Adding a Max-tier scenario family (the multi-day path)

A Max family is a triplet of YAML files:

1. `sre_gym/max/families/<family_id>.yaml` — family-level spec (topology, scenario_population, allowed_actions, reward_model, reference_instance, operator_notes)
2. `sre_gym/max/chaos/<family_id>_chaos_library.yaml` — composable chaos patterns
3. `sre_gym/max/compose/<family_id>.yaml` — docker-compose stack for the topology

The schema is documented in [`docs/MAX_TIER.md`](MAX_TIER.md) §2-§4 and concretely demonstrated in `ecommerce_vibecoded_saas`. Required design moves:

- **Pick a domain narrowly.** "E-commerce SaaS" is a topology; "general SaaS" is not. The chaos library, action set, and reward model all key off the domain shape.
- **Stub external dependencies.** Don't wire to real Stripe/Supabase; build stub servers with fault-injection toggles via env vars. This is what makes the Max tier *runnable in a sandboxed cluster* rather than dependent on third-party API quotas.
- **Specify operator_notes.** Cost estimate, isolation requirements, reset-safety guarantees. Without these, an enterprise SRE platform team can't evaluate whether to lift the family into a real cluster.
- **Constrain composition.** `composition_safety` declares always-safe pairs, unsafe pairs, and a max simultaneous patterns cap. Without it, two simultaneous gossip-cert expiries render the cluster unrecoverable.

---

## 4. Cross-tier compatibility

A new template added at the Basic tier doesn't automatically gain horizon-tier or realism-tier semantics. If you want a Basic template to *also* be an Advanced reference scenario, you write a separate YAML in `sre_gym/advanced/scenarios/` that wraps the Basic template's structure inside a multi-phase incident chain. This is intentional — the tier-specific shape is part of the tier's research claim, not derivable from the Basic template alone.

---

## 5. Tests for new scenarios

Every new template/scenario must have:

- **Smoke test:** baseline walk passes, score lands in expected band.
- **Wrong-action test:** rollback wrong service triggers `failure_type="wrong_remediation_target"`.
- **Premature-resolve test:** `declare_resolved` before checks pass triggers `failure_type="premature_resolution"`.
- **Noise-handling test:** querying a noise service reduces `noise_handling_score`.
- **Procgen test:** all 5 procgen variants resolve via the baseline path.

The test file template lives in [`unified_incident_env/tests/test_environment.py`](../unified_incident_env/tests/test_environment.py) — copy the relevant test, parametrize it on your new scenario_id, done.

---

## 6. Submission checklist

For a new Basic template to ship:

- [ ] Template dict added to `EXTRA_TEMPLATES` (or `_BASE_SCENARIOS`)
- [ ] Baseline lambda added to `extra_baselines()`
- [ ] `RootCauseType` enum extended in `models.py`
- [ ] 5 tests added (smoke, wrong-action, premature-resolve, noise-handling, procgen)
- [ ] `pytest unified_incident_env/tests -q` passes
- [ ] `python -m openenv.cli validate .` passes
- [ ] Scenario shows up at `GET /tasks` with all 5 procgen variants
- [ ] `train/data/` includes at least 3 trajectories of teacher-driven solves on the new template
- [ ] [`docs/BASIC_TIER.md`](BASIC_TIER.md) §1 table updated with the new template + skill description

For a new Advanced reference scenario:

- [ ] YAML file in `sre_gym/advanced/scenarios/`
- [ ] Loadable via `SREGym(tier=Tier.ADVANCED).list_scenarios()`
- [ ] [`docs/ADVANCED_TIER.md`](ADVANCED_TIER.md) §2 updated with one paragraph describing the scenario and what it tests

For a new Max family:

- [ ] Triplet of YAML files in `sre_gym/max/{families,chaos,compose}/`
- [ ] Loadable via `SREGym(tier=Tier.MAX).list_scenarios()`
- [ ] [`docs/MAX_TIER.md`](MAX_TIER.md) §2-4 updated with the family description, chaos patterns table, and operator notes
