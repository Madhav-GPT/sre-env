# Reward design

> Why the rubric is shaped, why the ceiling is hardened, and why the ≤ 0.80 baseline ceiling is a feature not a bug.

This document explains the reward decisions across all three tiers. The Basic-tier rubric is implemented in [`unified_incident_env/server/grader.py`](../unified_incident_env/server/grader.py); the Advanced/Max rubrics extend it with horizon-specific and realism-specific dimensions documented in their respective tier docs.

---

## 1. The core rubric (Basic tier, all 12 templates)

```
recovery_score        0.00 – 0.25      critical-path services healthy
containment_score     0.00 – 0.15      root cause removed (0.15) or service isolated (0.10)
verification_score    0.00 – 0.20      database_recovery (0.08) + end_to_end (0.12)
impact_score          0.00 – 0.05      user-impact reduced from baseline
efficiency_score      0.00 – 0.05      blast-radius budget preserved
speed_bonus           0.00 – 0.10      finishing under optimal_ticks (conditional on full verification)
noise_handling_score  0.00 – 0.05      penalizes querying distractor noise services
                      ----
total                 0.00 – 0.85      with public clamp to [0.01, 0.99]
```

Plus a per-tick *shaped* reward computed as the change in incident-health potential (see §3).

---

## 2. The five-component decomposition, defended

**Recovery (0.25)** is the highest-weighted dimension because it's the only one that's strictly necessary. An agent that doesn't restore service health doesn't matter how cleanly it executed everything else. The 0.25 cap is high enough that a partial recovery (degraded → degraded but better) earns partial credit, but not so high that an agent can ace recovery alone and skip verification.

**Containment (0.15)** rewards the *path* taken to get to recovery. A rollback that removes the cause earns 0.15; an isolation that reduces blast radius without removing the cause earns 0.10. Both are valid SRE moves, but the first is strictly better — and the rubric reflects that. The 0.05 gap is the price of choosing isolate-without-rollback as a permanent solution.

**Verification (0.20)** is split (0.08 + 0.12) so that the *end-to-end* check is worth more than the *database_recovery* check. End-to-end is the user-facing health gate. database_recovery is a per-component check that's necessary but not sufficient. An agent that runs only `database_recovery` and declares resolved leaves the gateway/worker path unverified — and the rubric penalizes that.

**Impact (0.05)** is small on purpose. User-impact reduction *follows from* recovery; double-counting it would distort the rubric toward "make the user-impact number go down" rather than "fix the underlying fault". The 0.05 cap acknowledges that customer-impact-aware operators *should* feel rewarded for it, without letting it dominate.

**Efficiency (0.05)** measures wasteful actions: redundant queries, repeated rollback attempts, isolating an already-isolated service. The 0.05 cap is again deliberate — being efficient matters, but not at the expense of being thorough. An agent that skips verification to save ticks scores worse on verification; an agent that double-queries doesn't lose much. The asymmetry is correct.

The 5-component decomposition is also what makes this a *composable rubric* in the OpenEnv framework sense: each dimension is independently tunable, independently testable (one test per dimension), and independently extensible (the Advanced tier *adds* dimensions; it doesn't *replace* the Basic ones).

---

## 3. The shaped per-tick reward

The grader returns a final score in [0, 0.85] only at episode end. Per-tick rewards are computed differently — they're *shaped* by the change in **incident-health potential**:

```
potential = 0.55 * sum(weight[s] * service_status_score[s]) for s in critical_services
          + 0.20 * (1 - user_impact)
          + 0.15 * (1 - slo_burn_rate)
          + 0.10 * containment_applied

per_tick_reward = -step_cost
                 + (potential_after - potential_before)
                 + bonus_from_action
                 - penalty_from_action
```

This gives dense intermediate signal: a correct rollback raises `potential` because services move toward healthy and `containment_applied` flips True. A wrong rollback or premature restart leaves potential flat (and pays an explicit penalty). Restarting the wrong service penalizes; restarting the right service after rollback raises potential.

**Why this matters for compute-bounded training.** Without dense shaping, GRPO at 800 steps on 60 scenarios doesn't converge in 12 hours of A100. We tested terminal-only rewards in early prototyping and watched the reward stay flat for 400+ steps before any signal emerged. With shaping, the reward starts climbing within the first 50 steps. The 12-hour budget is the design constraint; shaping is the technique that makes it feasible.

The shaping is *potential-based*: shaping rewards form a telescoping sum that exactly equals the difference in terminal potential, so the optimal policy under shaped rewards is identical to the optimal policy under unshaped rewards (Ng et al. 1999). That's a non-trivial property: it means shaping doesn't change *what* we're optimizing, only *how fast* the gradient finds the optimum.

---

## 4. The hardened ≤ 0.80 baseline ceiling

A scripted-optimal baseline that follows the canonical action sequence for each template tops out at ~0.77 across all 12 templates. The CI invariant `test_baseline_ceiling_is_hardened_below_080` enforces this — any reward-config change that pushes the baseline above 0.80 is rejected.

Why? Because **the headroom 0.80 → 0.99 is what GRPO trains into**. The baseline can earn:

- Full recovery (0.25)
- Full containment (0.15)
- Full verification (0.20)
- Full impact (0.05)
- Full efficiency (0.05)
- *Zero* speed bonus (it spends `optimal_ticks` exactly, no faster)
- Full noise handling (0.05) — the baseline avoids noise queries
- Hypothesis bonus partial (~0.06 for a calibrated hypothesis with high confidence)

That sums to ~0.81, then clamps to 0.80. To exceed 0.80, an agent must:

- Resolve in *fewer than* `optimal_ticks` (earns up to +0.10 speed bonus), AND
- Land a perfectly-calibrated high-confidence hypothesis on first try (earns up to +0.06)

A 3B specialist trained with GRPO learns to do both, because the dense shaping reward + group-relative advantages teach it which order of actions is fastest, and the explicit hypothesis-calibration reward teaches it to commit to a hypothesis with high confidence rather than hedge. Without the hardened ceiling, the env saturates at 0.85+ for a strong frontier model and there's no signal left to train against.

---

## 5. Penalty structure

Three negative-reward sources:

| Source | Magnitude | Triggers |
|---|---|---|
| `step_cost` | 0.01/tick | always (encourages efficiency) |
| `unsafe_action_penalty` | 0.08 (medium) / 0.12 (hard) | rollback wrong service, isolate wrong service |
| `premature_resolution_penalty` | 0.20 (medium) / 0.30 (hard) | `declare_resolved` before checks pass |
| `low_value_restart` (half-strength) | 0.04 (medium) / 0.06 (hard) | restart wrong service |
| `premature_restart` | 0.08 (medium) / 0.12 (hard) | restart correct service before cause removed |

The asymmetry between `low_value_restart` (half-penalty) and `premature_restart` (full-penalty) reflects the SRE judgment that "restarting the right thing too early" is worse than "restarting the wrong thing" — restarting too early *re-inherits* the bad state and resets progress, while restarting the wrong thing is just wasted action.

---

## 6. Hypothesis reward (anti-gaming)

`submit_hypothesis` is the action that scores the agent's *belief about the world*. It pays:

- `0.04` for correct root_cause (RootCauseType match against scenario truth)
- `0.03 × overlap` for affected_services overlap with truth
- `0.03 × quality` for recommended_next_action quality (bonus if right, -0.4 if wrong)
- `0.02 × calibration` for confidence calibration (bonus if confident-and-right, penalty if confident-and-wrong)

Total: up to ~0.12 per scenario. Critically, `submit_hypothesis` is **idempotent**: a second identical hypothesis returns 0 reward. An agent that spams hypotheses to fish for partial credit gets one shot per unique hypothesis.

Why this matters: in early prototyping a frontier-LLM agent gamed the reward by submitting 4 different hypotheses (one for each plausible cause) and harvesting partial credit on each. Idempotence kills that strategy.

---

## 7. The composable-rubric framework usage

The grader is structured so each dimension is computed independently:

```python
recovery_score        = compute_recovery(state, scenario)
containment_score     = compute_containment(state, scenario)
verification_score    = compute_verification(state, scenario)
impact_score          = compute_impact(state, scenario)
efficiency_score      = compute_efficiency(state, scenario)
speed_bonus           = compute_speed_bonus(state, scenario)
noise_handling_score  = compute_noise_handling(state, scenario)
final_score           = sum_and_clamp(...)
```

In OpenEnv-framework terms, this is the "composable rubric" pattern: each dimension is a small, testable, independently-tunable function. Adding the Advanced-tier `chained_incident_recognition` dimension is a 10-line patch — a new function, a new test, a new entry in the sum.

The alternative (a monolithic LLM-as-judge scoring the entire trajectory) is *cheaper to build* but *more expensive to debug*: when the score is wrong, you can't tell which dimension is broken. The composable approach surfaces failures at the right granularity.

---

## 8. Cross-tier reward shape

| Dimension | Basic | Advanced | Max |
|---|---|---|---|
| Core 7-dim rubric | ✅ | inherits | inherits |
| Hypothesis bonus | ✅ | ✅ | ✅ |
| Shaped per-tick reward | ✅ | ✅ | ✅ |
| `chained_incident_recognition` | — | ✅ | ✅ |
| `alternate_observability_use` | — | ✅ | ✅ |
| `pipeline_protection` | — | ✅ | ✅ |
| `containment_first` | — | ✅ | ✅ |
| `security_recognition` | — | ✅ | ✅ |
| `data_leak_window_documented` | — | ✅ | ✅ |
| `customer_comm_drafted` | — | ✅ | ✅ |
| `postmortem_quality` (rubric) | — | ✅ | — |
| `postmortem_quality` (learned critic) | — | — | ✅ |
| `actual_recovery_state` (binary) | — | — | ✅ |
| `revenue_lost_during_outage` | — | — | ✅ |
| `iac_remediation_applied` | — | — | ✅ |
| `runbook_update` | — | — | ✅ |
| `mttr` | — | — | ✅ |
| Outcome-scored | — | — | ✅ |

The reward shape *grows* monotonically across tiers — each tier strictly contains the previous tier's signals plus tier-specific additions. That means a model trained at Basic carries useful priors into Advanced training; an Advanced model carries useful priors into Max training. The tier escalation is a curriculum, not a benchmark substitution.

---

## 9. Test discipline

Every dimension has a corresponding test in `unified_incident_env/tests/`. The tests verify:

- The dimension is *independently zero* under appropriate conditions (e.g. `verification_score=0` when no checks have run)
- The dimension is *bounded* by its declared range (no off-by-one overflow)
- The dimension *reaches its cap* on at least one canonical trajectory (no dead-code dimensions)
- The CI invariant `baseline_ceiling ≤ 0.80` holds across all 12 templates

Adding a new dimension requires adding a test for each of those four properties. That's the cost of the composable-rubric design — and it's a cost we pay willingly because it's what makes the rubric debuggable.
