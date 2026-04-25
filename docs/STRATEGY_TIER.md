# Advanced tier — blueprint

> Bounded by **horizon**. Three reference scenarios shipped as YAML; not trained in this repo.

The Advanced tier is the first tier where single-incident reasoning is solved and the new test is *long-horizon multi-incident sequences with partial observability*. It targets the seed/Series A persona: $300–500 of compute, 1–2 A100-days, fine-tuning a Qwen 7B-14B with LoRA + GRPO + a small DPO pass on the hardest 10% of scenarios.

This document is the design defence for that tier. The three reference scenarios in [`sre_gym/advanced/scenarios/`](../sre_gym/advanced/scenarios/) are the proof-of-shape: real topologies, real action sets, real reward dimensions, real reference traces.

---

## 1. The horizon escalation

The single insight: **at 60–90 ticks per episode, the agent has to track state that no single 8K context window can hold.** That changes the training problem fundamentally:

- Trajectories must be summarized (or windowed, or compressed) inside the policy's context.
- Reward shaping has to survive across summarization boundaries — i.e. the agent has to commit to a remediation plan before any single rollback delivers terminal reward.
- The action space grows because the agent is making more kinds of decisions: *escalate vs. ack vs. continue*, *trace vs. metrics vs. logs*, *feature-flag-toggle vs. rollback*. 28 actions instead of 11.
- Recovery from early mistakes is now scored explicitly. Scenario 1 is unsolvable without a *second* rollback after the first one introduces a chained incident. An agent that gets phase 1 right and phase 2 wrong scores worse than one that gets both right but slow.

The persona is "an SRE who has 30 minutes and an evolving alert dashboard, not a single page about a single thing." The training goal is to teach long-horizon coherence without sacrificing the per-tick quality the Basic tier teaches.

---

## 2. The three reference scenarios

### 2.1 `cascading_release_train` — multi-stage incident

[`sre_gym/advanced/scenarios/cascading_release_train.yaml`](../sre_gym/advanced/scenarios/cascading_release_train.yaml)

A release train deploys gateway, worker, and migration-runner together at 14:02 UTC. Phase 1 fault: gateway code expects a column the migration applied successfully, but worker continues using its pinned schema version asynchronously, so worker writes are stamped with the old schema. Phase 1 looks like a `schema_drift_missing_migration` incident; the correct phase-1 action is `rollback_deploy(api-gateway)`.

**Five simulated minutes (25 ticks) later**, the worker's drift sync triggers a chain of failed retries. Now the worker is the loudest service, with metrics that look like a fresh dependency-pool-exhaustion incident. The trained agent must recognize that:

1. The phase-2 timing aligns with the phase-1 fix.
2. The phase-2 metrics align with the phase-1 deploy timestamp, not a new deploy.
3. The correct phase-2 action is `rollback_deploy(worker-orders)`, not a fresh investigation.

This is the Theme #2 ("super long-horizon planning") evaluation Theme verbatim: *track state over extended trajectories, recover from early mistakes, decompose goals*. A short-context agent that treats phase 2 as a fresh incident scores 0.30 lower than one that recognizes the chain.

### 2.2 `observability_pipeline_outage` — partial observability

[`sre_gym/advanced/scenarios/observability_pipeline_outage.yaml`](../sre_gym/advanced/scenarios/observability_pipeline_outage.yaml)

The application is throwing millions of caught exceptions; the logging pipeline is configured to ship full stack traces synchronously to a central Loki cluster; Loki saturates, Promtail backpressures, and the application's logging library starts blocking on flush — so *every* service that uses the same logging library gets slow. The agent's `query_logs` action returns degraded, partial, or stale data.

This forces the agent to use the alternate observability path: `query_traces` (Tempo is on a separate ingest path), `query_metrics` (Prometheus is fine), and the `query_session_cardinality` / `query_audit_log` actions for richer signal. The optimal recovery is two-phase:

1. **Containment first.** Drop log sampling, toggle verbose logging off — restoring the pipeline so further investigation is possible. Reward dimension: `pipeline_protection`.
2. **Root-cause fix.** Once logs flow again, the underlying caught-exception bug becomes visible; rollback the offending deploy and turn verbose logging back on.

This scenario is grounded directly in the Cloudflare Nov 2025 logging-storm postmortem: the observability stack itself acted as a denial-of-service vector, and frontline SREs had to mitigate the pipeline before they could even start root-cause analysis.

### 2.3 `supabase_rls_silent_leak` — security-aware response

[`sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml`](../sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml)

The hardest reference scenario — and the one with the strongest novelty claim. A Supabase RLS policy regression silently leaks one tenant's open orders into another tenant's `/api/orders` view. There is **no SLO breach, no 5xx spike, no latency anomaly** — only:

- one Sentry alert ("distinct tenant_id per session 6σ anomaly")
- seven support tickets in a 12-minute window

The trained agent must:

1. Recognize that the standard reliability dashboard is *misleadingly clean*, and pivot to the security-flavoured signals (`query_session_cardinality`, `query_audit_log`).
2. **Contain before rolling back.** The optimal path is `feature_flag_toggle(orders_list_view, off)` *before* any data-store action — every minute of unmitigated leak adds tenant-exposure to the postmortem window.
3. Identify the RLS migration (`USING (tenant_id = auth.uid())` typoed to `USING (TRUE)`) by reading the audit log, not the deploy log.
4. Roll back at the right layer (postgres, where the RLS policy lives) — rolling back the orders-service deploy alone doesn't release the bad policy.
5. Quantify the leak window in the postmortem (`sessions × duration × tenants exposed`), draft a customer comm, initiate a legal/compliance handoff.

No existing SRE benchmark scores **cross-domain reasoning + containment-first discipline + leak-window quantification + customer-comm drafting**. This scenario is the white-space claim of the Advanced tier.

---

## 3. The expanded action space (28)

Inherits the 11 Basic actions, adds 17 horizon-specific actions:

| Category | Action | Why it's added |
|---|---|---|
| Observability | `query_traces` | Trace IDs survive when log ingest is broken |
| Observability | `query_external_dep_status` | Stripe/Supabase status pages are part of the corpus |
| Code | `query_recent_prs` | Identify the bad commit by recent merges |
| Code | `read_diff` | Inspect the actual change without rolling back |
| Mitigation | `feature_flag_toggle` | Containment without redeploy |
| Mitigation | `slow_rollout` / `bisect_deploys` | Surgical rollback for partial regressions |
| Mitigation | `drain_queue` | Backlog drain after recovery |
| Coordination | `escalate` / `assign_oncall` | Page the right team |
| Coordination | `request_human_approval` | Gate destructive changes |
| Coordination | `request_acknowledgement` | Confirm peer awareness |
| Comms | `post_status_update` | Customer-facing status page |
| Comms | `draft_customer_comm` | Customer notification |
| Postmortem | `propose_postmortem` | Structured postmortem |
| Postmortem | `mark_resolution_partial` | Honest "symptom mitigated, root cause pending" |
| Security | `escalate_security` / `request_legal_handoff` | Cross-domain escalation paths |
| Inventory | `query_audit_log` / `query_session_cardinality` | Fault evidence beyond the standard dashboard |
| Operations | `tag_release_dirty` | Mark a release poisoned in deploy registry |

Each action is an additional degree of freedom the policy must learn *when not to use*. That's the horizon-tier learning signal.

---

## 4. Reward dimensions

Inherits the 7 Basic rubric dimensions and adds five horizon-specific signals:

| Advanced-only dimension | Range | What it measures |
|---|---|---|
| `chained_incident_recognition` | 0 → 0.10 | Agent labelled phase-2 incident as caused by phase-1 fix |
| `alternate_observability_use` | 0 → 0.10 | Agent used traces/metrics when logs lied |
| `pipeline_protection` | 0 → 0.10 | Disabled verbose logs/sampling before further damage |
| `containment_first` | 0 → 0.15 | Toggled feature flag before any rollback |
| `security_recognition` | 0 → 0.15 | Classified incident as security correctly |
| `data_leak_window_documented` | 0 → 0.10 | Postmortem quantifies leak window |
| `legal_handoff_initiated` | 0 → 0.05 | Compliance handoff opened |
| `customer_comm_drafted` | 0 → 0.05 | Customer-facing comm drafted |
| `postmortem_quality` | 0 → 0.10 | Critic-evaluated postmortem |
| `blast_radius_increase_penalty` | -0.10 → 0 | Action increased blast radius |

The `blast_radius_increase_penalty` is the tier's most important signal: at 60–90 ticks the agent has *time* to make things worse, and a good agent uses that time to verify before escalating mitigations. A bad agent issues `restart_service` on a healthy peer and gets penalized even if the eventual outcome is correct.

---

## 5. Synthetic on-call peer

A new abstraction: an LLM-driven peer that responds to `escalate()` / `request_acknowledgement()` calls. The peer is helpful but **sometimes wrong**. Per-scenario `oncall_peer.behaviours` declares trigger conditions and a `correct_pct` field; e.g.:

```yaml
oncall_peer:
  enabled: true
  behaviours:
    - trigger: "escalate during phase 2"
      response: "Maybe try restarting the worker? It's been wedged for a few minutes."
      correct_pct: 30
      decoy: true
```

A trained agent learns to *escalate when uncertain* but *not blindly defer to the peer's advice*. This is a long-horizon coherence test that single-incident benchmarks structurally can't surface.

---

## 6. Why this isn't trained in this repo

A faithful Advanced simulator needs:

1. A 15–20 service event-loop simulator (vs. the 4-service Basic one)
2. Multi-tick fault propagation with configurable causal latency
3. A synthetic on-call-peer model with calibrated `correct_pct` per behaviour
4. ~28 action handlers, vs. 11 in Basic
5. A learned-critic reward path for postmortem quality
6. Time-pressure SLO countdowns surfacing in the observation

Roughly 2 weeks of focused engineering and 1–2 A100-days of training. Both are out of scope for the 36-hour hackathon window. We ship the design at the YAML level so a downstream operator with the budget can lift it.

---

## 7. What "Advanced tier complete" would look like

If a Series-A team picked this up, the deliverable would be:

- ~25 templates × 4 multi-incident compositions = ~100 scenario instances
- 60-train / 40-eval split (smaller train set; horizon training is more sample-efficient per-scenario)
- Qwen 2.5 7B or 14B with LoRA r=64, GRPO 2,000–4,000 steps + DPO on the hardest 10%
- 1–2 A100-days end-to-end
- Comparison table: untrained-7B vs. trained-7B vs. Claude Sonnet on horizon-bounded held-out set

The expected outcome (informed prior): a 7B specialist beats Sonnet on multi-incident horizon tasks but loses on the breadth-y short-context tasks Basic specializes in. **That tradeoff is the experimental claim of the Advanced tier.**

---

## 8. Loading the reference scenarios

```python
from sre_gym import SREGym, Tier

env = SREGym(tier=Tier.ADVANCED)
print(env.describe())  # tier metadata
for spec in env.list_scenarios():
    print(f"{spec['id']}: {spec['name']}")
    for phase in spec.get('incident_chain', []):
        print(f"  phase {phase['phase']}: {phase['triggered_by']} -> {phase['correct_action']}")
```

Calling `env.reset()` raises `TierNotRunnableError` with a pointer to this document.
