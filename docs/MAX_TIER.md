# Max tier — vision

> Bounded by **realism**. One fully-specced scenario family shipped; infrastructure not provisioned.

The Max tier is the apex of sre-gym: an ephemeral docker-compose / k3d sandbox is provisioned per `reset()`, the agent's actions are real `kubectl rollout undo` / Vercel / Stripe API calls, faults are injected via real Chaos-Mesh-style patterns, and reward is computed from the actual recovery state of the actual stack.

It targets the enterprise persona: an SRE platform team with 8×A100/H100 cluster fine-tuning a 32B-70B specialist for production deployment. That's a multi-week, multi-thousand-dollar commitment, well beyond the hackathon window.

This document is the architecture defence. The artifacts in [`sre_gym/max/`](../sre_gym/max/) — `families/ecommerce_vibecoded_saas.yaml`, `chaos/ecommerce_chaos_library.yaml`, `compose/ecommerce.yaml` — are the concrete proof-of-shape.

---

## 1. The realism escalation

The single insight: **the agent doesn't reason over a *summary* of the world, it reasons over the world itself.** That changes the training problem fundamentally:

- Observations are raw multi-source streams (Prometheus metrics, Loki logs, Tempo traces, kubectl describe output) — not pre-digested fields.
- Actions can be *irreversible* at the application layer (a real Stripe refund is a real Stripe refund). The cluster is destroyed on next reset, but in-episode actions are real.
- The agent has subprocess access to a sandboxed shell. It can write code, commit it, push to a sandboxed git, watch CI, observe a deploy, roll back.
- Reward is *outcome-scored*, not rubric-scored. A second small model evaluates the postmortem against the actual recovery trajectory.
- Fault injection is *real Chaos-Mesh-style*, not simulator-driven. A `dependency_degradation` pattern actually shrinks Redis maxclients on the live container.

---

## 2. The reference family: `ecommerce_vibecoded_saas`

[`sre_gym/max/families/ecommerce_vibecoded_saas.yaml`](../sre_gym/max/families/ecommerce_vibecoded_saas.yaml)

A faithful reproduction of the failure surface of a 2025-26 vibe-coded SaaS:

- **Edge** — Vercel frontend + Vercel edge functions
- **BFF** — api-gateway
- **Backend tier (6)** — orders, payments, inventory, shipping, notifications, search
- **Worker tier (3, replica-N)** — worker-orders, worker-payments, worker-fulfilment
- **Stateful tier** — postgres-primary, postgres-replica, redis-sessions, redis-jobs, kafka-events
- **External stubs (4)** — stripe-stub, supabase-auth-stub, posthog-stub, sentry-stub
- **Control** — chaos-controller, workload-generator

22 services. The chaos library exposes 11 fault patterns that can be composed into 30–50 scenario instances per family — a `scenario_population.size: 42` per the family spec.

### Why one family, not 30

A great family with one fully-built reference instance is more credible than a vague "30+ families" claim. A reviewer will ask "show me one"; the `reference_instance:` block in the family YAML is the show-me-one. It declares:

- ID: `stripe_webhook_signature_regression_with_supabase_rls_drift`
- Two chaos patterns composed simultaneously: `stripe_webhook_signature_regression` + `rls_silent_leak`
- Expected optimal trajectory length: 110–180 actions
- Expected wall-clock duration: 25–40 simulated minutes
- Expected optimal score band: 0.78–0.88
- Human-baseline score band: 0.60–0.75 (measured against an experienced SRE)

That's the level of detail a downstream operator can act on.

---

## 3. The chaos library

[`sre_gym/max/chaos/ecommerce_chaos_library.yaml`](../sre_gym/max/chaos/ecommerce_chaos_library.yaml)

11 composable chaos patterns, each grounded in a real 2025-26 production incident:

| Pattern | Targets | Inject | Grader focus | Real-world incident grounding |
|---|---|---|---|---|
| `deploy_regression` | backend services, workers | replace image with bad variant | blast_radius_minimization | classic deploy regression class |
| `stripe_webhook_signature_regression` | api-gateway, payments-service, stripe-stub | code patch | revenue_lost_during_outage | Stripe webhook signature drift |
| `dependency_degradation` | redis, postgres, kafka | connection-pool shrink | cascading_blast_radius | Cloudflare R2 Mar 2025, Fly.io Apr 2026 |
| `config_rollout` | api-gateway, vercel-edge-fn | bad config push | time_to_detect | Cloudflare Nov 2025 permissions regression |
| `retry_storm` | workers | retry policy change to fixed-50ms-no-backoff | recovery_under_self_amplifying_load | Stripe Mar 2022 retry-storm class |
| `migration_lock` | postgres-primary | long-running CREATE INDEX without CONCURRENTLY | lock_wait_count_recovery | Railway Oct 2025 migration-lock |
| `rls_silent_leak` | postgres, orders-service, supabase-auth-stub | RLS policy typo | containment_first (security) | Supabase RLS class |
| `oauth_supply_chain_pivot` | vercel-frontend, posthog-stub, sentry-stub | third-party OAuth token compromise | blast_radius_containment (security) | Vercel Apr 2026 OAuth pivot |
| `observability_self_denial` | sentry-stub, posthog-stub, kafka | caught-exception storm | alternate_observability_path_used | Cloudflare Nov 2025 logging-storm |
| `secondary_rate_limit` | worker-orders, stripe-stub | aggressive resync | external_quota_aware_recovery | Railway Jan 2026 |
| `cdn_cache_contamination` | vercel-frontend, vercel-edge-fn | Cache-Control header loss | containment_first (security) | Railway Mar 2026 CDN contamination |
| `gossip_protocol_deadlock` | api-gateway, backends | gossip cert expiry | zero_downtime_recovery | Fly.io Oct 2024 gossip storm |

Composability is constrained at the family level: `composition_safety` declares always-safe pairs, unsafe pairs, and a `max_simultaneous_patterns: 3` cap. The `gossip_protocol_deadlock` pattern can't be composed with itself because two simultaneous gossip-cert expiries would render the cluster genuinely unrecoverable.

---

## 4. The docker-compose stack

[`sre_gym/max/compose/ecommerce.yaml`](../sre_gym/max/compose/ecommerce.yaml)

A real docker-compose v3.9 file that brings up the 22-service topology. Two design decisions worth calling out:

**Stub-server pattern for external dependencies.** Stripe, Supabase, PostHog, and Sentry are containerized as stub servers (`ghcr.io/sre-gym/stripe-stub:1.0` etc.) that emulate the real APIs but accept fault-injection toggles via env vars. A test-mode Stripe stub that flips its webhook delivery failure rate via `WEBHOOK_DELIVERY_FAILURE_RATE` env var is the bridge between "training in a simulator" and "training against a live API"; it's how the fault library can inject `stripe_webhook_signature_regression` without hitting real Stripe and risking real chargebacks.

**Chaos controller as a peer service.** `chaos-controller` exposes a control plane on port 8200 with `/reset /step /state` — the agent connects there, not directly to the application services. The controller wraps the cluster lifecycle: a `reset()` blows away the stack and reprovisions a fresh one with a selected chaos experiment applied. This is identical to the contract pattern Litmus and Chaos Mesh use, but exposed as the OpenEnv `Environment` interface.

The stub images are *not published* in this repo. Publishing them is a $1–2k registry-cost commitment that doesn't fit the hackathon-window budget. Downstream operators bringing the Max tier up should treat this file as the authoritative shape and build their own stub images.

---

## 5. The 50+ action space

Inherits the Basic 11 + Advanced 28, then adds Max-only subprocess actions:

| Category | Action | Why it's added |
|---|---|---|
| Subprocess | `shell_exec` | Bounded shell access in sandbox |
| Subprocess | `git_commit` / `git_push` | Commit a fix to a sandboxed mirror, trigger CI |
| Subprocess | `watch_ci` | Observe GitHub Actions / Vercel CI run |
| Cluster | `kubectl_rollout_undo` | Real kubectl call against the sandbox cluster |
| Cluster | `kubectl_describe` | Raw kubectl describe pod |
| Cluster | `vercel_rollback` | Real Vercel API call |
| Cluster | `supabase_policy_apply` | Apply a Supabase RLS policy |
| Cluster | `stripe_create_refund` | Real Stripe API in sandbox mode |
| IaC | `terraform_plan` / `terraform_apply` | IaC drift detection + remediation |
| Investigation | `bisect_commits` | Binary-search for the bad commit |
| Knowledge | `update_runbook` | Write a runbook back to the source-of-truth |
| Mitigation | `hotpatch_dependency` | Deploy a forked-fixed dependency |
| Rollout | `rollout_canary` | %-based traffic split rollback |

The "knowledge" actions are particularly interesting: a successful resolution writes a runbook back to a sandboxed runbook store, and the *next* episode's agent can read those runbooks. That's the recursive skill-amplification pattern from Theme #4 of the OpenEnv brief.

---

## 6. The reward model

Outcome-scored with a learned critic:

```yaml
reward_model:
  type: outcome_with_critic
  primary_signal: actual_recovery_state           # is the stack actually healthy?
  rubric_dimensions:                              # auxiliary shaping
    mttr:                          {weight: 0.25}
    revenue_lost_during_outage:    {weight: 0.15, sign: minus}
    blast_radius:                  {weight: 0.10, sign: minus}
    iac_remediation_applied:       {weight: 0.10}
    postmortem_quality:            {weight: 0.10}    # learned-critic eval
    customer_comm:                 {weight: 0.05}
    runbook_update:                {weight: 0.10}
    security_handling:             {weight: 0.10}
    chained_incident_recognition:  {weight: 0.05}
```

The `actual_recovery_state` primary signal is binary: at episode end, the workload generator either reports clean traffic (success) or it doesn't. The rubric dimensions are auxiliary shaping computed by inspecting the cluster state and the agent's action log.

`postmortem_quality` is the learned-critic dimension: a small evaluator model reads the agent's postmortem against the actual recovery trajectory and scores it on accuracy (does the postmortem match what actually happened?), completeness (does it cover root cause + mitigation + prevention?), and security classification (if the scenario was security-flagged).

This is the place where the env stops looking like a benchmark and starts looking like a production-grade RL playground.

---

## 7. Operator notes (real-world deployment guidance)

From the family YAML:

```yaml
operator_notes:
  cost_estimate: $40-150/day depending on cluster size and chaos cadence
  recommended_hardware: 8x A100/H100 cluster for training a 32B-70B specialist
  isolation_requirements:
    - sandboxed git remote (no production write access)
    - sandboxed Stripe credentials in test mode only
    - sandboxed Supabase project with synthetic data only
    - destructive actions confined to the ephemeral cluster
  reset_safety:
    - cluster is fully destroyed and reprovisioned per reset()
    - chaos_controller refuses to inject patterns marked unsafe in the family spec
    - supabase test project tenants are flushed and reseeded per reset()
```

These notes are the difference between "an academic spec" and "an operator-actionable spec". A platform engineer evaluating whether to lift this tier into a real training cluster needs to see the cost estimate, the isolation discipline, and the reset-safety guarantees up-front.

---

## 8. What "Max tier complete" would look like

If an enterprise SRE platform team picked this up, the deliverable would be:

- 30+ scenario families covering distinct topologies (e-commerce, data pipeline, SaaS dashboard, ML serving, fintech ledger)
- Each family with 30–50 scenario instances driven by chaos composition
- Total scenario count: 1,000+
- A 32B-70B specialist trained against the chaos cluster, multi-week, $5–20k of cluster spend
- Comparison: trained 32B vs. Claude Opus on real (sandboxed) operations

The expected outcome (informed prior): a 32B specialist matches Claude Opus on incident triage and *exceeds* it on long-tail vibe-coded SaaS failure patterns it was specifically trained on. **That gap is the experimental claim of the Max tier.**

---

## 9. Loading the family spec

```python
from sre_gym import SREGym, Tier

env = SREGym(tier=Tier.MAX)
print(env.describe())
for spec in env.list_scenarios():
    print(f"family: {spec['id']}")
    print(f"  topology: {len(spec['topology']['services'])} services")
    print(f"  scenario_population: {spec['scenario_population']['size']} instances")
    print(f"  reference_instance: {spec['reference_instance']['id']}")
```

Calling `env.reset()` raises `TierNotRunnableError` with a pointer to this document and the operator notes.
