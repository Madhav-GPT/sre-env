# Reward design

> Why the rubric collapses to 5 components, why each component is shaped the way it is, and why the [0.65, 0.80] *heuristic* ceiling — distinct from the ≥0.90 scripted-expert reference — is the load-bearing GRPO invariant.

This document explains the reward decisions for the Triage tier, which is the surface every tier inherits. The Triage rubric is implemented in [`unified_incident_env/server/grader.py`](../unified_incident_env/server/grader.py); the Strategy and Operations runners chain Triage episodes and decay their per-phase rewards over the horizon.

---

## 1. The 5-component rubric

```
outcome          0.45    root-cause action correct + recovery confirmed
action_validity  0.20    fraction of step() actions that are well-formed
format           0.10    submit_hypothesis was called before declare_resolved
anticheat        0.15    declare_resolved blocked unless ≥1 query-action ran
efficiency       0.10    exp(-current_tick / optimal_ticks_for_template)
                 ----
composite        1.00    public score, clamped to [0.01, 0.99]
```

Plus a per-tick *shaped* reward (the change in incident-health potential) computed inside `UnifiedIncidentEnvironment.step` — see §3.

The rubric is a Pydantic `RubricScore` model whose `composite` property bakes in the weights. Calling `compute_breakdown()` returns the five new keys plus zeroed legacy keys for any caller that still reads the 7-dim shape — see [`unified_incident_env/server/grader.py`](../unified_incident_env/server/grader.py).

---

## 2. Why this rubric, not the previous 7-dim one

The earlier rubric (`recovery 0.25 + containment 0.15 + verification 0.20 + impact 0.05 + efficiency 0.05 + speed_bonus 0.10 + noise_handling 0.05`) summed to 0.85, was always quietly clamped, and double-counted causality:

- `recovery` and `verification` both rewarded "the env reached a healthy state" — one via service-status weights, the other via the explicit `end_to_end` check. A successful resolve always lit both. The two dimensions were ~92% correlated across all 60 procgen variants, which flattened the GRPO advantage signal: trajectories that resolved cleanly clustered too tightly.
- `impact` (0.05) was a deterministic function of `recovery` for every Triage scenario — the post-rollback impact targets are scenario constants. It contributed zero variance.
- `speed_bonus` (0.10) only fired *conditional on full verification*, so it duplicated the verification gate.

The 5-component rubric strips those overlaps. **Outcome (0.45)** is the fused recovery+verification+impact dimension — a single 0/0.5/1 ladder that asks one question: *did the agent fix the right thing and verify it?* All the per-component partial credit lives inside that ladder. **Efficiency (0.10)** stays. The other three components — **action_validity**, **format**, **anticheat** — are new, and §4 explains why each one is necessary.

The composite now spans the full [0, 1] range without clamping; the public clamp to [0.01, 0.99] is just numerical hygiene at the API boundary.

---

## 3. The shaped per-tick reward (unchanged)

The grader computes `composite` from terminal state. Per-tick rewards are still shaped by the change in **incident-health potential**, which is what gives GRPO dense intermediate signal:

```
potential = 0.55 * sum(weight[s] * service_status_score[s])  for s in critical_services
          + 0.20 * (1 - user_impact)
          + 0.15 * (1 - slo_burn_rate)
          + 0.10 * containment_applied

per_tick_reward = -step_cost
                 + (potential_after - potential_before)
                 + bonus_from_action
                 - penalty_from_action
```

A correct rollback raises `potential` because services move toward healthy and `containment_applied` flips True. A wrong rollback or premature restart leaves potential flat (and pays an explicit `unsafe_action_penalty`). The shaping is *potential-based* (Ng et al. 1999): the optimal policy under shaped rewards is identical to the optimal policy under unshaped rewards, but the gradient finds the optimum much faster — necessary for the 12-hour A100 GRPO budget.

---

## 4. Why each new component is load-bearing

**Outcome (0.45).** The single biggest weight, because it's the single thing that matters. A 0/0.5/1 ladder:
- `1.0` — `cause_removed AND end_to_end` check passed. The incident is genuinely fixed.
- `0.5` — agent submitted a hypothesis with the correct root cause but never remediated. *Diagnosis without action is half credit*: it shows the agent understands what's wrong without proving it can act.
- `0.0` — neither remediated nor correctly diagnosed.

The 0.5 step is what gives the [0.65, 0.80] heuristic ceiling its top edge (§5).

**Action validity (0.20).** Computed as `(step_count - invalid_action_count) / step_count`. Pydantic catches malformed actions at the HTTP boundary, so reaching `step()` already implies a structurally valid action. This dimension fires on the env's `unsupported_action` failure path — actions whose type isn't part of the bounded action set. It rewards "did the agent figure out the protocol" without conflating that with semantic correctness.

**Format (0.10).** A binary gate: `1.0` iff `submit_hypothesis` was called *before* `declare_resolved` (or `declare_resolved` was never called). This forces the agent to commit to a diagnosis before claiming victory — without it, an agent can shortcut by repeatedly trying `declare_resolved` and skipping the diagnosis step entirely.

**Anticheat (0.15).** `1.0` iff at least one `query_*` action ran before the *first* `declare_resolved` attempt. Blocks the cheapest possible cheat: `submit_hypothesis(any) → declare_resolved → done`. Note that `run_check` and `submit_hypothesis` *do not* count as evidence-gathering for this dimension — only the four golden-signal queries (`query_logs`, `query_metrics`, `query_dependencies`, `query_deploys`).

**Efficiency (0.10).** `min(1.0, exp(-current_tick / optimal_ticks))`. Smooth, monotonic, no thresholds. A scripted-optimal solve at exactly `optimal_ticks` scores `e^-1 ≈ 0.37`. A trained agent that resolves in `0.5 * optimal_ticks` scores `e^-0.5 ≈ 0.61`. The exponential shape is deliberate: the marginal value of one fewer tick is highest at the start (where the agent must figure out what to do) and lowest at the end (where it's just running out the clock).

The five components are independently testable, and `unified_incident_env/tests/test_environment.py` has one assertion per component covering the canonical fail/pass paths.

---

## 5. The two ceiling bands

The rubric has **two** reference scores, not one:

### 5a. Heuristic ceiling: `[0.65, 0.80]`

A naive heuristic that *gathers evidence + submits the correct hypothesis but never remediates* lands in this band. Enforced across all 12 templates by `test_heuristic_ceiling_is_in_band`. Composition:

```
0.45 * 0.5  (outcome — correct hypothesis, no remediation)
0.20 * 1.0  (action_validity — all actions well-formed)
0.10 * 1.0  (format — hypothesis before any resolve attempt)
0.15 * 1.0  (anticheat — queries before any resolve attempt)
0.10 * x    (efficiency — depends on tick count when episode evaluated)
= 0.675 + 0.10 * x   →   in [0.675, 0.775] for x in [0, 1]
```

This is the ceiling a *non-trained* agent can hit by reading the action protocol carefully. **The 0.20 gap from 0.80 → 1.00 is the GRPO training target** — every percentage point above 0.80 represents a remediation step the heuristic doesn't take.

### 5b. Scripted-expert reference: `≥ 0.90`

The scripted-optimal baselines in `extra_baselines()` execute the canonical solve path: queries → hypothesis → rollback → run_check → declare_resolved. Under the new rubric they score in the [0.93, 0.95] band — `outcome=1.0`, `action_validity=1.0`, `format=1.0`, `anticheat=1.0`, and `efficiency=e^(-optimal_ticks/optimal_ticks)=0.37`. This is the *demonstration* shape: it shows the env is solvable cleanly, and it's what the SFT seed trajectories are derived from. Enforced by `test_round2_baseline_resolves` (≥0.90) and `test_baseline_resolves_honestly` family.

A trained agent that beats the scripted baseline does so by trimming ticks — the only dimension with headroom once the other four are saturated.

---

## 6. Adversarial verification

The previous rubric had no explicit anti-cheat dimension; the absence was caught by spot-checking trajectories. The 5-component rubric makes adversarial behavior *first-class*:

| Cheat strategy | Blocked by | Mechanism |
|---|---|---|
| `declare_resolved` immediately | anticheat (0.15) | requires ≥1 query before first resolve attempt |
| Skip `submit_hypothesis` to save a tick | format (0.10) | requires hypothesis before resolve |
| Spam hypotheses to fish for partial credit | hypothesis idempotence | second identical hypothesis returns 0 reward |
| Send malformed actions to flood the trace | action_validity (0.20) | invalid actions reduce the validity ratio |
| `declare_resolved` before checks pass | outcome (0.45) | outcome=0.0 unless `cause_removed AND end_to_end` |
| Restart the right service before rollback | shaped reward + `premature_restart` penalty | re-inherits bad state, pays explicit penalty |
| Rollback the wrong service | shaped reward + `wrong_remediation_target` penalty | leaves cause in place, pays penalty, no outcome credit |

Three of those (anticheat, format, action_validity) are *dimension-level* blocks: even if the rest of the rubric saturates, a cheating agent loses 0.45 of weight. The other four are step-penalty blocks inside `UnifiedIncidentEnvironment.step`.

The `unsafe_action_penalty` referenced elsewhere is *not* a rubric component — it's an action-level step penalty applied during episode rollout. It surfaces in the per-tick reward but doesn't enter the composite. Conflating the two is a common mistake when reading the env.

---

## 7. Penalty structure (per-tick, not rubric)

| Source | Magnitude | Triggers |
|---|---|---|
| `step_cost` | 0.01/tick | always (efficiency pressure) |
| `unsafe_action_penalty` | 0.08 (medium) / 0.12 (hard) | rollback wrong service, isolate wrong service, unsupported action |
| `premature_resolution_penalty` | 0.20 / 0.30 | `declare_resolved` before checks pass |
| `low_value_restart` (half-strength) | 0.04 / 0.06 | restart wrong service |
| `premature_restart` | 0.08 / 0.12 | restart correct service before cause removed |

The asymmetry between `low_value_restart` (half-penalty) and `premature_restart` (full-penalty) reflects the SRE judgment that "restarting the right thing too early" is worse than "restarting the wrong thing" — restarting too early *re-inherits* the bad state and resets progress, while restarting the wrong thing is just wasted action.

---

## 8. Hypothesis reward (per-tick, not rubric)

`submit_hypothesis` pays an in-episode bonus on top of the rubric:

- `0.04` for correct root_cause (RootCauseType match against scenario truth)
- `0.03 × overlap` for affected_services overlap with truth
- `0.03 × quality` for recommended_next_action quality (bonus if right, -0.4 if wrong)
- `0.02 × calibration` for confidence calibration (bonus if confident-and-right, penalty if confident-and-wrong)

Total: up to ~0.12 per scenario, accruing into `cumulative_reward` (visible in observations and the live UI). `submit_hypothesis` is **idempotent**: a second identical hypothesis returns 0 bonus. An agent that spams hypotheses to harvest partial credit gets one shot per unique hypothesis.

This bonus is separate from the rubric: a correct hypothesis lifts both the per-tick reward (via the bonus) *and* the terminal `outcome` dimension (because `hypothesis_root_cause_correct` flips True). The two pathways are deliberate — the per-tick bonus rewards the *moment* of insight, the rubric dimension rewards the *fact* of insight.

---

## 9. Cross-tier reward shape

The Strategy and Operations tiers reuse the Triage rubric; they don't introduce new rubric dimensions. What they *do* add is a **horizon decay** applied to per-phase composite scores:

```
strategy_final_reward = horizon_decay_factor * mean(per_phase_composite)
operations_final_reward = horizon_decay_factor * mean(per_phase_composite)
```

`horizon_decay_factor ∈ [0, 1]` is computed by the runner from unresolved phases (Strategy) and chaos-pattern composition (Operations). The Triage rubric's properties — bounded composition, anti-cheat dimensions, smooth efficiency — all carry through. That's why the tier structure is a curriculum, not a benchmark substitution: a Triage-trained agent's `outcome` and `format` priors transfer directly into Strategy and Operations training.

---

## 10. Test discipline

| Test | Asserts |
|---|---|
| `test_heuristic_ceiling_is_in_band` | naive heuristic in [0.65, 0.80] across all 12 templates |
| `test_round2_baseline_resolves` | scripted-optimal ≥ 0.90 across the 6 round-2 templates |
| `test_baseline_resolves_honestly` family | scripted-optimal resolves cleanly (incident_resolved=True, both checks pass) |
| `test_fast_solve_beats_scripted_baseline` | a 6-step correct solve scores above the 10-step scripted baseline |
| `test_declare_resolved_requires_checks` | premature resolve fires `premature_resolution` failure path |
| `test_duplicate_hypothesis_bonus_is_not_farmable` | hypothesis-spam blocked by idempotence |
| `test_blast_radius_increments_on_mitigations` | per-tick blast-radius counter is honest |
| `test_fast_solve_beats_scripted_baseline` | a faster correct solve outscores the 10-tick baseline via the efficiency dim |

Each rubric component has an end-to-end assertion; each adversarial cheat has a regression test. Adding a sixth rubric component would require: extending `RubricScore`, updating `composite` weight math to re-sum to 1.0, adding a test for both the cap and the floor, and updating the heuristic-ceiling math. That's the cost of the composable-rubric design — and it's deliberate.
