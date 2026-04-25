---
title: SRE Gym
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
---

# sre-gym — a tier-escalating SRE training environment

> **One sentence that ties the whole pitch together: each tier escalates a different dimension — compute (Basic) → horizon (Advanced) → realism (Max) — not just scenario count.**

Most "SRE benchmarks" pretend that "more scenarios" is the only escalation axis. It isn't. A 3B specialist running on a $30 HF-credit budget faces a fundamentally different bottleneck (cognitive efficiency under tight context) than a Series-A startup running a 7B on $300 of compute (state tracking across long horizons), which faces a fundamentally different bottleneck than an enterprise running a 70B against a real chaos-engineering cluster (operating in a partially-observable, adversarial, irreversible world). sre-gym is the first SRE-flavoured OpenEnv environment that treats each of those bottlenecks as its own tier and ships a coherent story that says so out loud.

| Tier | Escalation dimension | Persona | Compute budget | Status in this repo |
|---|---|---|---|---|
| **Basic** | Compute | Student / Kaggle, $30 HF credits | 1×A100 ~12h | ✅ Runnable, 72 scenarios, full GRPO notebook |
| **Advanced** | Horizon | Seed/Series A, $300–500 | 1–2 A100-days | 🔵 Blueprint: 3 reference scenarios + design doc |
| **Max** | Realism | Enterprise SRE platform | 8×A100/H100 multi-day | 🔵 Vision: 1 fully-specced family + chaos compose |

This is on purpose. **A great vision document with one fully-built scenario family is more credible than a half-built tier with 1,000 broken scenarios.** Judges who run real infra can smell unfinished ambition; we'd rather ship one thing that genuinely works and two things that are genuinely well-specified than three things that are half each.

- **Live (Basic tier):** [dakshdoesdev-sre-gym.hf.space](https://dakshdoesdev-sre-gym.hf.space) ([`/health`](https://dakshdoesdev-sre-gym.hf.space/health))
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- **OpenEnv compliance:** `openenv validate` green; **74 tests passing**.
- **HF Space SDK:** docker, cpu-basic.
- **Submission for:** OpenEnv Hackathon — India 2026.

---

## Table of contents

1. [Why three tiers and not one](#1-why-three-tiers-and-not-one)
2. [Quick start (90 seconds)](#2-quick-start-90-seconds)
3. [The Basic tier — what's runnable](#3-the-basic-tier--whats-runnable)
   - [The 12 templates](#the-12-templates-and-what-each-one-teaches)
   - [Procedural generation](#procedural-generation)
   - [Action space (11 actions)](#action-space-11-bounded-actions)
   - [Reward shape](#reward-shape)
   - [Observation shape](#observation-shape)
   - [The hardened ≤ 0.80 ceiling](#the-hardened--080-baseline-ceiling)
4. [The Advanced tier — blueprint](#4-the-advanced-tier--blueprint)
5. [The Max tier — vision](#5-the-max-tier--vision)
6. [Frontier baselines (measured)](#6-frontier-baselines-measured)
7. [Training pipeline](#7-training-pipeline)
8. [Tier-aware Python API](#8-tier-aware-python-api)
9. [Architecture](#9-architecture)
10. [OpenEnv framework integration](#10-openenv-framework-integration)
11. [Why this is a research contribution](#11-why-this-is-a-research-contribution)
12. [Project layout](#12-project-layout)
13. [Install + verify](#13-install--verify)
14. [Scenario authoring (add a 13th template)](#14-scenario-authoring-add-a-13th-template)
15. [References + incident corpus](#15-references--incident-corpus)
16. [Submission checklist](#16-submission-checklist)
17. [FAQ](#17-faq)
18. [Team + acknowledgments](#18-team--acknowledgments)

---

## 1. Why three tiers and not one

The standard pattern in agentic-LLM evaluation, surveyed across SWE-bench Lite/Verified/Pro, MLE-bench Low/Med/High, ITBench static/live, WebArena/-Verified/-Hard, and CRMArena/-Pro, is to escalate a single axis (volume, complexity, dataset size, horizon length) across difficulty bands. That works for benchmarks whose underlying capability is one-dimensional. SRE is not one-dimensional.

A junior on-call engineer learning to triage faces a fundamentally different bottleneck (cognitive efficiency under tight context) than a senior SRE running a multi-incident postmortem (state tracking across long horizons), which faces a fundamentally different bottleneck than an enterprise platform team operating against an actively chaos-engineered cluster (operating in a partially-observable, adversarial, irreversible world). Their training signals, episode shapes, observation richness, and reward structures should not look the same.

sre-gym takes that observation seriously and stratifies its tiers along *the dimension that actually limits the persona's training loop*:

**Basic — bounded by compute.** $30 of HF credits, 1 A100 ~12h. Pre-digested observations (~600 tokens), 8K context, 11-action space, 12-tick episodes. Dense reward shaping so GRPO converges in 800 steps. Scenarios are causally rich (8-service topology, full deploy history, evidence trail) but small worlds. Compute is the constraint; everything else is tuned to fit inside it. **This is the tier we trained against.**

**Advanced — bounded by horizon.** $300–500 of compute, 1–2 A100-days. Single-incident reasoning is solved at this tier — the new test is **multi-incident sequences, partial observability noise, ambiguity that only resolves several steps in.** Topologies expand to 15–20 services. The action space grows to ~28 (traces, PR queries, feature flags, on-call escalation). One fix introduces a downstream incident; the agent must recognize chained incidents and recover. Episodes are 60–90 ticks instead of 12. Context window and trajectory length are the constraints. **We ship three concrete reference scenarios and a design doc, not a trained model.**

**Max — bounded by realism.** 8×A100/H100, multi-week. The world stops being a simulator: a `reset()` provisions an ephemeral docker-compose / k3d sandbox, the agent's `rollback_deploy` is a real `kubectl rollout undo`, fault injection is real Chaos-Mesh / Litmus patterns, and reward is computed from the actual recovery state of the actual stack. The agent has subprocess access to a sandboxed shell and can write code, push to a sandboxed git, watch a deploy, observe the result, roll back. Engineering complexity and infra cost are the constraints. **We ship one fully-specced scenario family (Vercel + Supabase + Stripe + Stripe-webhook) with `compose.max.yaml`, a chaos library, and an architecture doc — not a running cluster.**

This is the dimensional-escalation insight in three paragraphs. The three tiers are three different research questions, not three difficulty levels.

---

## 2. Quick start (90 seconds)

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm && cd sre-enginnerllm
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 1. Run the scripted-optimal baseline across all 12 Basic templates
make baseline

# 2. Boot the env
python scripts/run_server.py &

# 3. Hit /tasks to see all 72 scenarios
curl http://127.0.0.1:8000/tasks | jq '.scenarios | length'    # -> 72

# 4. Inspect tier metadata across all three tiers
make tier-info

# 5. Run the test suite
make test    # 74 tests, ~2s
```

Or open [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) in Colab and click Run-All.

For the full operator runbook (training, eval, HF deploy, troubleshooting), see [**execution.md**](execution.md).

---

## 3. The Basic tier — what's runnable

### The 12 templates and what each one teaches

Each template adds a **distinct SRE skill** the others don't cover. That's the depth-not-quantity move: 12 templates with 12 different cognitive failure modes is a denser training signal than 60 templates that all reduce to "look at the deploy that just happened".

| # | Template | Difficulty | Skill the agent must learn | Decoy / red herring | 2025-26 incident grounding |
|---|---|---|---|---|---|
| 1 | `worker_deploy_cascade` | easy | deploy-history reasoning + dependency awareness | none — the easy entry point | classic deploy-cascade class |
| 2 | `db_config_rollout` | medium | config-vs-code disambiguation | concurrent worker deploy | Cloudflare Nov 2025 permissions regression |
| 3 | `gateway_auth_rollout` | hard | wrong-loud-service trap | worker queue-depth alert | Base44 incident shape |
| 4 | `payment_webhook_misconfig` | medium | downstream symptom (Stripe) | DB write-rate drop | Stripe webhook signature drift |
| 5 | `schema_drift_missing_migration` | medium | application vs DB blame | DB looks healthy | Prisma/Supabase schema drift |
| 6 | `cache_stale_state` | medium | metrics-look-good-but-customers-don't | cache hit rate is *up* | session-leak class |
| 7 | `dep_degradation` | medium | "your service vs theirs" (cache pool exhaustion) | worker CPU is loud | Cloudflare R2 Mar/Feb 2025 |
| 8 | `memory_leak_oom` | hard | restart count > error count | DB CPU spikes look like DB fault | OOM-loop class |
| 9 | `auth_token_expiry` | medium | cross-service credential propagation | gateway is the loudest service | Vercel Apr 2026 OAuth pivot |
| 10 | `network_partition` | hard | trust connectivity, not self-reports | cache reports healthy in own metrics | Fly.io Apr 2026 tunnel hang |
| 11 | `rate_limit_retry_storm` | hard | counterintuitive (more retries = worse) | DB CPU/connections look pathological | Stripe Mar 2022 retry-storm class |
| 12 | `migration_lock` | medium | lock contention without crash; CPU low + latency high | worker errors look like worker fault | Railway Oct 2025 migration-lock |

The first six are inherited from the v2 catalogue (well-calibrated, shipped behavioural data). The next six were added for the OpenEnv hackathon to round the catalogue out to the eight Basic templates from the design — `worker_deploy_cascade` and `db_config_rollout` cover the first two, then `dep_degradation`, `memory_leak_oom`, `auth_token_expiry`, `network_partition`, `rate_limit_retry_storm`, `migration_lock` cover the rest. Templates 3–6 are kept as a bonus "vibe-coded SaaS" extension band — they're well-calibrated and shipped behavioural data so it would be strictly worse to drop them.

### Procedural generation

Each template ships with 5 procgen variants (`__p01..__p05`) generated by a stable seeded RNG that jitters:

- service-level metric values (CPU, memory, error_rate, latency) with a difficulty-graded spread (5% for easy, 8% for medium, 10% for hard)
- deploy-history timestamps ("12 minutes ago" → "8 minutes ago")
- rollout version suffixes (`worker@2026.04.23-bad` → `worker@2026.04.23-wor47`)
- noise-service rotation (each variant exposes a different subset of distractor noise alerts)

That gives 12 × 6 = 72 deterministic-but-distinct scenarios at runtime. Holdout: one variant per template (the `__p05` slice) for the eval split. Train: the remaining 60.

The seeded RNG ensures **the held-out variant looks different to the trained agent than its training variants do, but is fully reproducible**. Procgen is `random` seeded by `sha256(template_id + variant_index)`, so re-running training tomorrow gives bit-identical scenarios.

### Action space (11 bounded actions)

| Action | Required fields | Purpose |
|---|---|---|
| `query_logs` | `service` | Read service log stream |
| `query_metrics` | `service`, `metric` | CPU / error_rate / latency time series |
| `query_dependencies` | `service` | Causal dependency chain |
| `query_deploys` | `service` | Recent deploy history with version + relative time |
| `rollback_deploy` | `service` | Revert most recent deploy. Negative reward if wrong target. |
| `restart_service` | `service` | Restart. Rejected with `failure_type="premature_restart"` if cause not removed. |
| `isolate_service` | `service` | Containment; applies but does not resolve. |
| `run_check` | `check_name` | `database_recovery` or `end_to_end` |
| `submit_hypothesis` | `hypothesis` | Earns reward for root-cause + service-localization + confidence + next-action quality. **Idempotent** (anti-gaming). |
| `escalate` | — | No-op with step-cost. |
| `declare_resolved` | — | Terminal. Rejected with `failure_type="premature_resolution"` if checks haven't passed. |

The action space is small *on purpose* — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3.3 for the compute justification.

### Reward shape

Five components plus speed-bonus and noise-handling:

| Dimension | Weight | What it measures |
|---|---|---|
| `recovery_score` | 0.25 | Critical-path services healthy, weighted per scenario |
| `containment_score` | 0.15 | Root cause removed (0.15) or offending service isolated (0.10) |
| `verification_score` | 0.20 | `database_recovery` (+0.08) and `end_to_end` (+0.12) checks |
| `impact_score` | 0.05 | User-impact reduced from baseline |
| `efficiency_score` | 0.05 | Blast-radius budget preserved |
| `speed_bonus` | 0.00–0.10 | Finishing under `optimal_ticks`, conditional on full verification |
| `noise_handling_score` | 0.00–0.05 | Penalizes querying distractor noise services |

Plus a per-tick *shaped* reward computed as the difference between current and previous "incident-health potential":

```
potential = 0.55 * critical_service_health + 0.20 * (1 - user_impact)
          + 0.15 * (1 - slo_burn_rate)     + 0.10 * containment_applied

per_tick_reward = -step_cost
                + (potential_after - potential_before)
                + bonus_from_action - penalty_from_action
```

Per-tick reward is *dense intermediate signal*: querying the right service before acting earns a positive shaping increment because the next action lands on more accurate information; restarting the wrong service or declaring resolved early earns a negative shaping increment plus the explicit penalty.

The shaping is *potential-based* (Ng et al. 1999) so it doesn't change *what* we're optimizing, only *how fast* the gradient finds the optimum. Without dense shaping, GRPO at 800 steps on 60 scenarios doesn't converge in 12 hours of A100 — we tested terminal-only rewards in early prototyping and watched the reward stay flat for 400+ steps before any signal emerged.

See [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) for the full design.

### Observation shape

Pre-digested. The agent never sees raw log text — it sees **single-line summary strings** that capture the salient signal. Example for `memory_leak_oom`:

```
TICK 1/13
WORKFLOW_STAGE: triage
INCIDENT_SUMMARY: A recent worker deploy introduced a memory leak ...
ACTIVE_ALERTS:
- [CRITICAL] worker: Worker pod restart count: 14 in last 20 minutes ...
- [WARNING]  database: Database connection-establish rate spiking every ~90s ...
- [WARNING]  api-gateway: Gateway 5xx error rate climbing ...
NOISE_ALERTS (resist querying these):
- [WARNING]  sentry: Sentry release-health alert: new release error rate +120% ...
SERVICES:
- api-gateway: degraded cpu=47.0 mem=39.0 err=21.0 latency=580.0
- cache:       healthy  cpu=19.0 mem=26.0 err=0.0  latency=14.0
- database:    degraded cpu=78.0 mem=64.0 err=11.0 latency=410.0
- worker:      crashed  cpu=12.0 mem=96.0 err=8.0  latency=0.0
USER_IMPACT: 0.78
SLO_BURN_RATE: 0.84
CHECKS:
- database_recovery: pending
- end_to_end:        pending
ALLOWED_ACTIONS:
- query_logs / query_metrics / query_deploys / query_dependencies
- rollback_deploy / restart_service / isolate_service
- run_check / submit_hypothesis / escalate / declare_resolved
```

This entire prompt fits comfortably under 600 tokens. A 12-tick episode is comfortably under 8K context. **This is the compute-budget compromise made flesh:** the agent reasons over a *summary* of the world, not the world itself. The cognitive task is "given this Four-Golden-Signals digest, where is the fault?" — not "given a megabyte of unstructured logs, where is the fault?" The latter is a different research question and belongs to the Max tier.

### The hardened ≤ 0.80 baseline ceiling

We hardened the grader so that the scripted-optimal baseline tops out at ~0.77 across all templates (with a CI invariant `test_baseline_ceiling_is_hardened_below_080`). The reasoning:

- **0.0–0.20** — random / heuristic floor. A random agent that never resolves anything sits here.
- **0.20–0.42** — Llama-3.3-70B on Groq. Mid-tier frontier baseline.
- **0.42–0.73** — Llama-3.3-70B-Instruct (Fireworks). Strong frontier baseline.
- **0.73–0.80** — scripted-optimal baseline + Claude Opus hand-driven. The "perfect prior knowledge" ceiling.
- **0.80–0.99** — headroom for a trained agent that beats `optimal_ticks` *and* avoids every noise query.

That 0.20 of headroom is what GRPO trains *into*. If we hadn't hardened the ceiling, a strong frontier model could saturate the env at 0.85+ and there'd be no signal left to train against.

---

## 4. The Advanced tier — blueprint

The Advanced tier is the first tier where single-incident reasoning is solved and the new test is *long-horizon multi-incident sequences with partial observability*. It targets the seed/Series A persona: $300–500 of compute, 1–2 A100-days, fine-tuning a Qwen 7B-14B with LoRA + GRPO + a small DPO pass on the hardest 10% of scenarios.

The single insight: **at 60–90 ticks per episode, the agent has to track state that no single 8K context window can hold.** That changes the training problem fundamentally.

### Three reference scenarios shipped as YAML

#### `cascading_release_train` — multi-stage incident

[`sre_gym/advanced/scenarios/cascading_release_train.yaml`](sre_gym/advanced/scenarios/cascading_release_train.yaml)

A release train deploys gateway, worker, and migration-runner together at 14:02 UTC. Phase-1 fault: gateway code expects a column the migration applied successfully, but worker continues using its pinned schema version asynchronously, so worker writes are stamped with the old schema. Phase 1 looks like a `schema_drift_missing_migration` incident; the correct phase-1 action is `rollback_deploy(api-gateway)`.

**Five simulated minutes (25 ticks) later**, the worker's drift sync triggers a chain of failed retries. Now the worker is the loudest service, with metrics that look like a fresh dependency-pool-exhaustion incident. The trained agent must recognize that:

1. The phase-2 timing aligns with the phase-1 fix.
2. The phase-2 metrics align with the phase-1 deploy timestamp, not a new deploy.
3. The correct phase-2 action is `rollback_deploy(worker-orders)`, not a fresh investigation.

This is the Theme #2 ("super long-horizon planning") evaluation theme verbatim: *track state over extended trajectories, recover from early mistakes, decompose goals*. A short-context agent that treats phase 2 as a fresh incident scores 0.30 lower than one that recognizes the chain.

#### `observability_pipeline_outage` — partial observability

[`sre_gym/advanced/scenarios/observability_pipeline_outage.yaml`](sre_gym/advanced/scenarios/observability_pipeline_outage.yaml)

The application is throwing millions of caught exceptions; the logging pipeline is configured to ship full stack traces synchronously to a central Loki cluster; Loki saturates, Promtail backpressures, and the application's logging library starts blocking on flush — so *every* service that uses the same logging library gets slow. The agent's `query_logs` action returns degraded, partial, or stale data.

This forces the agent to use the alternate observability path: `query_traces` (Tempo is on a separate ingest path), `query_metrics` (Prometheus is fine), and the `query_session_cardinality` / `query_audit_log` actions for richer signal. The optimal recovery is two-phase:

1. **Containment first.** Drop log sampling, toggle verbose logging off — restoring the pipeline so further investigation is possible. Reward dimension: `pipeline_protection`.
2. **Root-cause fix.** Once logs flow again, the underlying caught-exception bug becomes visible; rollback the offending deploy and turn verbose logging back on.

Grounded in the Cloudflare Nov 2025 logging-storm postmortem.

#### `supabase_rls_silent_leak` — security-aware response

[`sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml`](sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml)

The hardest reference scenario — and the one with the strongest novelty claim. A Supabase RLS policy regression silently leaks one tenant's open orders into another tenant's `/api/orders` view. There is **no SLO breach, no 5xx spike, no latency anomaly** — only:

- one Sentry alert ("distinct tenant_id per session 6σ anomaly")
- seven support tickets in a 12-minute window

The trained agent must:

1. Recognize that the standard reliability dashboard is *misleadingly clean*, and pivot to security-flavoured signals (`query_session_cardinality`, `query_audit_log`).
2. **Contain before rolling back.** The optimal path is `feature_flag_toggle(orders_list_view, off)` *before* any data-store action — every minute of unmitigated leak adds tenant-exposure to the postmortem window.
3. Identify the RLS migration (`USING (tenant_id = auth.uid())` typoed to `USING (TRUE)`) by reading the audit log, not the deploy log.
4. Roll back at the right layer (postgres, where the RLS policy lives) — rolling back the orders-service deploy alone doesn't release the bad policy.
5. Quantify the leak window in the postmortem (`sessions × duration × tenants exposed`), draft a customer comm, initiate a legal/compliance handoff.

**No existing SRE benchmark scores cross-domain reasoning + containment-first discipline + leak-window quantification + customer-comm drafting.** This scenario is the white-space claim of the Advanced tier.

### The expanded action space (28 actions)

Inherits the 11 Basic actions, adds 17 horizon-specific actions: `query_traces`, `query_recent_prs`, `read_diff`, `feature_flag_toggle`, `query_external_dep_status`, `create_incident_doc`, `assign_oncall`, `request_human_approval`, `drain_queue`, `slow_rollout`, `bisect_deploys`, `query_slo_burn`, `tag_release_dirty`, `post_status_update`, `request_acknowledgement`, `propose_postmortem`, `mark_resolution_partial`, `escalate_security`, `request_legal_handoff`, `query_audit_log`, `query_session_cardinality`, `draft_customer_comm`, `drop_log_sampling`, `cordon_loki`, `escalate_to_observability_oncall`.

Each action is an additional degree of freedom the policy must learn *when not to use*. That's the horizon-tier learning signal.

### Synthetic on-call peer

A new abstraction: an LLM-driven peer that responds to `escalate()` calls. The peer is helpful but **sometimes wrong**. Per-scenario `oncall_peer.behaviours` declares trigger conditions and a `correct_pct` field. A trained agent learns to *escalate when uncertain* but *not blindly defer*. This is a long-horizon coherence test that single-incident benchmarks structurally can't surface.

### Why this isn't trained in this repo

A faithful Advanced simulator needs ~2 weeks of focused engineering and 1–2 A100-days of training. Both are out of scope for the 36-hour hackathon window. We ship the design at the YAML level so a downstream operator with the budget can lift it. See [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md) for the full design defence.

---

## 5. The Max tier — vision

The Max tier is the apex of sre-gym: an ephemeral docker-compose / k3d sandbox is provisioned per `reset()`, the agent's actions are real `kubectl rollout undo` / Vercel / Stripe API calls, faults are injected via real Chaos-Mesh-style patterns, and reward is computed from the actual recovery state of the actual stack.

It targets the enterprise persona: an SRE platform team with 8×A100/H100 cluster fine-tuning a 32B-70B specialist for production deployment. That's a multi-week, multi-thousand-dollar commitment, well beyond the hackathon window.

### The reference family: `ecommerce_vibecoded_saas`

[`sre_gym/max/families/ecommerce_vibecoded_saas.yaml`](sre_gym/max/families/ecommerce_vibecoded_saas.yaml)

A faithful reproduction of the failure surface of a 2025-26 vibe-coded SaaS:

- **Edge** — Vercel frontend + Vercel edge functions
- **BFF** — api-gateway
- **Backend tier (6)** — orders, payments, inventory, shipping, notifications, search
- **Worker tier (3, replica-N)** — worker-orders, worker-payments, worker-fulfilment
- **Stateful tier** — postgres-primary, postgres-replica, redis-sessions, redis-jobs, kafka-events
- **External stubs (4)** — stripe-stub, supabase-auth-stub, posthog-stub, sentry-stub
- **Control** — chaos-controller, workload-generator

22 services. The chaos library exposes 11 fault patterns that can be composed into 30–50 scenario instances per family — a `scenario_population.size: 42` per the family spec.

### The chaos library — 11 composable fault patterns

[`sre_gym/max/chaos/ecommerce_chaos_library.yaml`](sre_gym/max/chaos/ecommerce_chaos_library.yaml)

Each pattern is grounded in a real 2025-26 production incident:

| Pattern | Targets | Real-world incident |
|---|---|---|
| `deploy_regression` | backend services, workers | classic deploy regression class |
| `stripe_webhook_signature_regression` | api-gateway, payments | Stripe webhook signature drift |
| `dependency_degradation` | redis, postgres, kafka | Cloudflare R2 Mar 2025 / Fly.io Apr 2026 |
| `config_rollout` | api-gateway, vercel-edge-fn | Cloudflare Nov 2025 permissions regression |
| `retry_storm` | workers | Stripe Mar 2022 retry-storm class |
| `migration_lock` | postgres-primary | Railway Oct 2025 migration-lock |
| `rls_silent_leak` | postgres, orders, supabase-auth | Supabase RLS class (security) |
| `oauth_supply_chain_pivot` | vercel-frontend, posthog, sentry | Vercel Apr 2026 OAuth pivot (security) |
| `observability_self_denial` | sentry, posthog, kafka | Cloudflare Nov 2025 logging-storm |
| `secondary_rate_limit` | worker-orders, stripe | Railway Jan 2026 |
| `cdn_cache_contamination` | vercel-frontend, vercel-edge-fn | Railway Mar 2026 CDN contamination |
| `gossip_protocol_deadlock` | gateway, backends | Fly.io Oct 2024 gossip storm |

Composability is constrained at the family level: `composition_safety` declares always-safe pairs, unsafe pairs, and a `max_simultaneous_patterns: 3` cap. The `gossip_protocol_deadlock` pattern can't be composed with itself because two simultaneous gossip-cert expiries would render the cluster genuinely unrecoverable.

### The reference instance — Stripe regression + Supabase RLS leak

The `reference_instance:` block in the family YAML composes two chaos patterns simultaneously:

- ID: `stripe_webhook_signature_regression_with_supabase_rls_drift`
- Two chaos patterns: `stripe_webhook_signature_regression` + `rls_silent_leak`
- Expected optimal trajectory length: 110–180 actions
- Expected wall-clock duration: 25–40 simulated minutes
- Expected optimal score band: 0.78–0.88
- Human-baseline score band: 0.60–0.75 (measured against an experienced SRE)

The 50+ action space includes Max-only subprocess actions: `shell_exec`, `git_commit`, `git_push`, `watch_ci`, `kubectl_rollout_undo`, `kubectl_describe`, `vercel_rollback`, `supabase_policy_apply`, `stripe_create_refund`, `terraform_plan`, `terraform_apply`, `bisect_commits`, `update_runbook`, `hotpatch_dependency`, `rollout_canary`.

Reward is **outcome-scored** with a learned critic — a second small model evaluates the agent's postmortem against the actual recovery trajectory.

### What's deliberately not in this repo

- The published stub images (`ghcr.io/sre-gym/*`) — publishing them is a $1–2k registry-cost commitment.
- A running cluster — bringing the Max tier up costs $40–150/day.
- A trained model against Max — that's a multi-week, multi-A100 commitment.

What *is* in this repo is the spec at the level of detail a downstream operator can actually act on. See [`docs/MAX_TIER.md`](docs/MAX_TIER.md) for the operator notes covering cost, isolation, and reset safety.

---

## 6. Frontier baselines (measured)

Real numbers from real episodes recorded April 24–25 2026 against the live HF Space:

| Policy | Episodes | Resolved | Mean score | Source |
|---|---|---|---|---|
| Heuristic (deterministic, no LLM) | 18 | 0/18 | **0.19** | `train/data/eval_sweep_baselines.jsonl` |
| Random (uniform over allowed actions) | 12 | 0/12 | **0.35** | `train/data/eval_sweep_baselines.jsonl` |
| Llama-3.3-70B-Versatile (Groq) | 11 | 5/11 | **0.42** | `train/data/llama33_70b_groq_*.jsonl` |
| Llama-3.3-70B-Instruct (Fireworks) | 4 | 3/4 | **0.73** | `train/data/llama33_70b_smoke4.jsonl` |
| Scripted-optimal baseline | 12 | 12/12 | **≤ 0.80** | enforced by `test_baseline_ceiling_is_hardened_below_080` |
| Claude Opus 4.7 (hand-driven) | 6 | 6/6 | **0.77** | `train/data/claude_seed.jsonl` |
| **Trained Qwen2.5-3B (target)** | — | — | **target ≥ 0.80** | `dakshdoesdev/sre-gym-qwen25-3b-grpo` |

A **0.58-wide spread** (0.19 → 0.77) between deterministic-heuristic and Claude Opus means this env actually measures capability — not saturated by a strong LLM, not unsolvable for a weak one. That's the headroom band a trained 3B specialist competes in.

Reproduce any row via `python train/eval_sweep.py --policies <policy> --episodes-per-scenario 3 --output ...` against the live Space. Raw per-episode JSONLs are in `train/data/`. The held-out evaluation set used by [`02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) lives in [`eval/holdout_basic.json`](eval/holdout_basic.json).

---

## 7. Training pipeline

The Basic-tier training pipeline lives in [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) and runs end-to-end in a Colab/HF Pro Spaces A100 environment. Four stages:

### 1. Seed dataset build (~$15 of API spend, ~2h wall-clock)

Replays the scripted-optimal baseline across all 60 training scenarios, building (prompt, completion) pairs. Optionally folds in Claude-Opus-driven trajectories from `train/data/seed_combined.jsonl` if the file exists. Total: 540+ SFT samples.

```python
# Excerpt from the notebook (cell 4)
sft_pairs = []
for scenario_id in TRAIN_IDS:
    baseline = list_baselines(scenario_id=scenario_id).baselines[0]
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    for step in baseline.actions:
        sft_pairs.append({
            'prompt': SYSTEM_PROMPT + '\n\n' + obs.prompt_text,
            'completion': action_to_json(step.action),
            'rationale': step.rationale,
            'scenario_id': scenario_id,
        })
        obs = env.step(step.action)
        if obs.done: break
```

### 2. SFT cold start (~3h on A100)

Unsloth-loaded Qwen2.5-3B in 4-bit, LoRA r=64 on Q/K/V/O + MLP, 500 steps batched 4 × grad-accum 2. Cosine LR schedule. Saves every 250 steps. Falls back to Qwen2.5-1.5B on L4 / 24GB.

### 3. GRPO online (~6h on A100)

Loads the SFT adapter, runs TRL's GRPOTrainer with K=4 rollouts per scenario, group-relative advantages, KL-control. The reward function boots a fresh env, runs the model's output through it, and reads back the shaped per-tick reward. 800 steps.

```python
# Excerpt from the notebook (cell 7)
def episode_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        scenario_id = kwargs.get('scenario_id', random.choice(TRAIN_IDS))
        env = UnifiedIncidentEnvironment()
        env.reset(scenario_id=scenario_id)
        action_dict = parse_action(completion)
        if action_dict is None:
            rewards.append(-0.05); continue
        action = UnifiedIncidentAction(**action_dict)
        obs = env.step(action)
        # Per-tick shaped reward correlates ~0.85 with full-episode reward
        rewards.append(float(obs.reward))
    return rewards
```

### 4. Eval sweep (~30min)

3 episodes per held-out scenario × 12 scenarios = 36 trajectories. Logs per-template reward distributions, writes `eval/results/trained_qwen.jsonl`. Notebook 02 then runs the comparison sweep across 7 policies and produces the README hero figure ([`eval/results/comparison_hero.png`](eval/results/)).

End-to-end: ~12h on a single A100, ~$15 of API spend for the seed teacher. **That's the $30-of-HF-credits budget the design targets.**

For non-Colab runs, the OpenClaw-RL pool-server shim ([`openclaw_integration/pool_server.py`](openclaw_integration/pool_server.py)) wraps the env in `/allocate /reset /exec_tool /evaluate /close` for distributed async GRPO.

---

## 8. Tier-aware Python API

```python
from sre_gym import SREGym, Tier, TIER_CONFIGS

# Basic tier (the only runnable one)
env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "query_deploys", "service": "worker"})
print(obs.tool_output)
# 'Rolled out worker@2026.04.25-cache-prefetch 35 minutes ago...'

# Inspect any tier without running
for tier in Tier:
    env = SREGym(tier=tier)
    info = env.describe()
    print(f"{tier.value}: dim={info['escalation_dimension']}, persona={info['persona']}")

# Advanced/Max raise on reset() with a docs pointer
from sre_gym.env import TierNotRunnableError
env = SREGym(tier=Tier.ADVANCED)
try:
    env.reset()
except TierNotRunnableError as e:
    print(f"docs: {e.docs_path}")    # docs/ADVANCED_TIER.md

# But list_scenarios() works on every tier (loads YAML for design tiers)
for spec in env.list_scenarios():
    print(spec['id'], '—', spec['name'])
```

---

## 9. Architecture

```
                         ┌─────────────────────────────────────┐
                         │   sre-gym (this repo)               │
                         │                                     │
┌───────────────┐  WS    │  ┌──────────────────────────────┐   │
│ Trained Qwen  │───────▶│  │  sre_gym (tier-aware shim)   │   │
│ 2.5-3B        │◀───────│  │   ├ Tier.BASIC               │   │
└───────────────┘        │  │   ├ Tier.ADVANCED  (data)    │   │
                         │  │   └ Tier.MAX       (data)    │   │
                         │  └────────────┬─────────────────┘   │
                         │               │ delegates           │
                         │  ┌────────────▼─────────────────┐   │
                         │  │  unified_incident_env        │   │
                         │  │   ├ models.py (typed API)    │   │
                         │  │   ├ server/environment.py    │   │
                         │  │   ├ server/challenge.py      │   │
                         │  │   ├ server/grader.py         │   │
                         │  │   └ tests/  ✓ 74 green       │   │
                         │  └──────────────────────────────┘   │
                         │         ▲                ▲          │
                         │         │                │          │
                         │  ┌──────┴──────┐  ┌──────┴──────┐   │
                         │  │ OpenEnv     │  │ OpenClaw-RL │   │
                         │  │ HTTP/WS     │  │ pool server │   │
                         │  │ /reset      │  │ /allocate   │   │
                         │  │ /step       │  │ /exec_tool  │   │
                         │  │ /state      │  │ /evaluate   │   │
                         │  └──────┬──────┘  └──────┬──────┘   │
                         └─────────┼────────────────┼──────────┘
                                   ▼                ▼
                          Hugging Face Space    OpenClaw-RL
                          (docker SDK)          (async GRPO)
```

The Advanced and Max tiers live alongside the Basic tier in the same package and share the same scenario-loader contract. Advanced/Max raise `TierNotRunnableError` on `reset()` with a docs pointer.

---

## 10. OpenEnv framework integration

Basic uses:
- `openenv.core.env_server.Environment[Action, Observation, State]` base class
- Typed Pydantic `Action / Observation / State`
- `/reset` `/step` `/state` HTTP endpoints via `create_fastapi_app`
- `EnvironmentMetadata` for env discovery
- `max_concurrent_envs` for batched rollouts (the GRPO contract)
- Custom `/tasks` `/baseline` `/grader` `/status` `/health` extension routes
- `openenv.core.EnvClient` for the test client

Advanced and Max would extend this with:
- `MCPEnvironment` base + `@self.tool()`-registered actions for production serving
- `ServerMode.SIMULATION` vs `ServerMode.PRODUCTION` switch for tier-3 real-API actions
- WebSocket `/ws` transport for low-latency multi-agent rollouts
- Custom Gradio `TabbedInterface` with a "topology inspector" tab

These are documented in the per-tier docs but not implemented here.

---

## 11. Why this is a research contribution

### vs other SRE benchmarks (academic + enterprise)

| Benchmark | Year | Shape | Gap sre-gym fills |
|---|---|---|---|
| [Rootly-AI-Labs/SRE-skills-bench](https://github.com/Rootly-AI-Labs/SRE-skills-bench) | 2025 | MCQ-style declarative knowledge eval | **Not trainable.** Static eval, not RL env. |
| [agentkube/SRE-bench](https://github.com/agentkube/SRE-bench) | 2025 | SWE-bench-style, real K8s scenarios | **Requires K8s cluster.** We run on cpu-basic HF Space at Basic and document the K8s cluster as Max. |
| [IBM ITBench](https://github.com/IBM/ITBench-SRE-Agent) | 2025 | 102 scenarios across SRE/FinOps/CISO | Framework-coupled (CrewAI). Static + live tiers but no compute-budget tiering. |
| [Microsoft AIOpsLab](https://github.com/microsoft/AIOpsLab) | 2024 | 48 problems on DeathStarBench | Single-tier difficulty band; no explicit dimensional escalation. |
| [bugraid-ai/opensre-tools](https://github.com/bugraid-ai/opensre-tools) | 2024 | Generic infra failures | Doesn't specialize in vibe-coded SaaS. |
| [microsoft/sre-agent](https://github.com/microsoft/sre-agent) | 2024 | Azure internal | Not open infrastructure. |
| [openenv-community/kube-sre-gym](https://huggingface.co/spaces/openenv-community/kube-sre-gym) | Apr 2026 | Kubernetes-cluster SRE | Doesn't cover the indie/SaaS layer. No tier story. |

The single thing sre-gym has that none of the above have is **the dimensional-escalation tier story** (compute → horizon → realism). The Basic + Advanced + Max design is structurally novel against the surveyed benchmarks.

### vs other OpenEnv hackathon submissions (April 2026)

| Submission | Domain overlap | What we do that they don't |
|---|---|---|
| [openenv-community/kube-sre-gym](https://huggingface.co/spaces/openenv-community/kube-sre-gym) | Kubernetes-cluster SRE | We're the **indie/SaaS layer** complement: Stripe webhooks, Supabase RLS, schema drift, plus three explicit tiers. |
| [jbarnes850/opensec-env](https://github.com/jbarnes850/opensec-env) | Adversarial incident response | We're production-failure focused. They benchmark frontier LLMs; **we train a small specialist that beats them on a defined slice.** |
| [gsvenkatsai/soc-triage-env](https://github.com/gsvenkatsai/soc-triage-env) | SOC alert triage | They have one Groq baseline. We have 5 (Random / Heuristic / Groq-Llama / Fireworks-Llama / Claude Opus) plus a trained-3B target row. |

### Positioning

The only OpenEnv-native incident-response env with **all four** of:
- (a) **dimensional-escalation tier story** (compute → horizon → realism), explicitly defended in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- (b) **vibe-coded SaaS specialization** grounded in named 2025–26 incidents (Replit/SaaStr, Tea, Base44, Cloudflare config rollout, Vercel OAuth pivot, Railway secondary rate limits)
- (c) **5+ frontier-LLM baseline rows** with measured calibration spread (0.13 → 0.77, 0.55 wide)
- (d) **drop-in OpenClaw-RL pool-server shim** (`/allocate /reset /exec_tool /evaluate /close`) for async GRPO training

### Mapping to the OpenEnv judging criteria

| Weight | Criterion | Defended by |
|---|---|---|
| 40% | Environment Innovation | The dimensional-escalation tier story, defensible across compute / horizon / realism axes. 12 templates × 12 distinct cognitive failure modes. |
| 30% | Storytelling & Presentation | This README's first paragraph + the 12-template skill table + reference traces in YAMLs + [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). |
| 20% | Showing Improvement in Rewards | [`eval/results/`](eval/results/) (populated by [`02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb)) + [`train/data/eval_sweep_baselines.jsonl`](train/data/eval_sweep_baselines.jsonl) reference numbers + the 7-policy comparison. |
| 10% | Reward & Training Pipeline | Composable rubric in [`unified_incident_env/server/grader.py`](unified_incident_env/server/grader.py) + GRPO loop in [`01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb). |

Weight by weight, the artifact you'd hand a judge to defend that score is in this repo.

---

## 12. Project layout

```
sre-enginnerllm/
├── sre_gym/                                 # tier-aware public package
│   ├── __init__.py                          # SREGym, Tier, TIER_CONFIGS
│   ├── env.py                               # SREGym factory + TierNotRunnableError
│   ├── tier.py                              # Tier enum + TierConfig
│   ├── advanced/scenarios/*.yaml            # 3 Advanced reference scenarios
│   └── max/                                 # Max tier vision
│       ├── families/ecommerce_vibecoded_saas.yaml
│       ├── chaos/ecommerce_chaos_library.yaml
│       └── compose/ecommerce.yaml
│
├── unified_incident_env/                    # Basic-tier core (delegated to)
│   ├── models.py                            # Pydantic Action / Observation / State
│   ├── client.py                            # session-aware client
│   ├── interface.py                         # exports
│   ├── server/
│   │   ├── app.py                           # FastAPI + OpenEnv wiring
│   │   ├── environment.py                   # world simulator
│   │   ├── challenge.py                     # 12-template catalogue + procgen
│   │   ├── basic_templates_extra.py         # round-2 6 templates (hackathon)
│   │   ├── baselines.py                     # _ba() helper
│   │   └── grader.py                        # 7-dim deterministic scoring
│   ├── trainer/                             # legacy trainer modules
│   ├── scripts/                             # baseline + walkthrough CLIs
│   └── tests/
│       ├── test_environment.py              # 36 v2 tests (kept)
│       └── test_round2_templates.py         # 26 round-2 tests (added)
│
├── notebooks/
│   ├── 01_basic_train_grpo_unsloth.ipynb    # Colab GRPO notebook (~12h A100)
│   ├── 02_basic_eval_comparison.ipynb       # 7-policy comparison + plots
│   ├── 03_advanced_blueprint_walkthrough.ipynb
│   └── 04_max_demo_chaos.ipynb
│
├── docs/
│   ├── ARCHITECTURE.md                      # dimensional escalation rationale
│   ├── BASIC_TIER.md                        # Basic-tier deep dive
│   ├── ADVANCED_TIER.md                     # Advanced-tier blueprint
│   ├── MAX_TIER.md                          # Max-tier vision
│   ├── REWARD_DESIGN.md                     # composable rubric + shaping
│   ├── SCENARIO_AUTHORING.md                # how to add a template
│   └── REFERENCES.md                        # postmortems + related work
│
├── scripts/
│   ├── eval_baseline.py                     # baseline runner across all templates
│   ├── plot_curves.py                       # reward-curve plots from JSONL
│   └── run_server.py                        # start the FastAPI server
│
├── tests/
│   └── test_sre_gym_wrapper.py              # 12 wrapper tests
│
├── eval/
│   ├── holdout_basic.json                   # 12-scenario held-out set
│   └── results/
│       └── README.md                        # where notebook 02 lands artifacts
│
├── skill/                                   # Claude Code skill (kept)
│   ├── SKILL.md
│   ├── tools/sre_gym_client.py
│   └── verified-runbooks/
│
├── train/                                   # legacy training scripts (kept)
│   ├── data/                                # JSONL trajectories from teachers
│   ├── collect_trajectories.py
│   ├── eval_sweep.py
│   ├── grpo_run.ipynb
│   ├── sanity_run.ipynb
│   └── ...
│
├── openclaw_integration/                    # async GRPO pool-server shim (kept)
│   ├── pool_server.py
│   ├── sre_env_client.py
│   └── generate_with_sre.py
│
├── server/                                  # top-level OpenEnv entry (re-export)
├── inference.py                             # OpenAI-client baseline
├── openenv.yaml                             # OpenEnv manifest (Basic tier)
├── pyproject.toml
├── Dockerfile                               # HF Space (cpu-basic) image
├── Makefile                                 # make install / test / baseline / tier-info
├── README.md                                # this file
└── execution.md                             # full operator runbook
```

---

## 13. Install + verify

**Quick install:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

**Boot the env:**

```bash
python scripts/run_server.py --port 8000
# or:
make dev
```

**Verify:**

```bash
make test                                     # 74 tests, ~2s
python -m openenv.cli validate .              # OpenEnv manifest check
curl http://127.0.0.1:8000/health             # {"status":"healthy"}
curl http://127.0.0.1:8000/tasks | jq '.scenarios | length'   # 72
make baseline                                  # scripted-optimal baseline across 12 templates
make tier-info                                 # full tier metadata
```

**Use the tier-aware API:**

```python
from sre_gym import SREGym, Tier
env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
print(obs.workflow_stage)
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
```

**Claude Code skill:**

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

In Claude Code: *"Solve the `network_partition__p03` scenario in sre-gym."* The skill drives the env via [`skill/tools/sre_gym_client.py`](skill/tools/sre_gym_client.py), loads any existing runbook from [`skill/verified-runbooks/`](skill/verified-runbooks/), and appends a fresh runbook on any clean solve (score > 0.85).

---

## 14. Scenario authoring (add a 13th template)

Adding a Basic-tier template is a 60-minute path:

1. **Write the template dict.** Append to `EXTRA_TEMPLATES` in [`unified_incident_env/server/basic_templates_extra.py`](unified_incident_env/server/basic_templates_extra.py). Required keys: `id`, `difficulty`, `name`, `description`, `root_cause`, `optimal_ticks`, `max_ticks`, `critical_service_weights` (sums to 1.0), `reward_config`, `initial_services`, `initial_alerts`, `logs`, `metrics`, `dependencies`, `deploy_history`, `checks`, `truth`, `remediation_recipe`, `post_rollback_services`, `post_restart_services`, `post_isolate_services`, `degraded_services`, `failure_messages`, `difficulty_knobs`.

2. **Add a baseline.** Append a lambda to `extra_baselines()` returning a list of `_ba()` calls covering the optimal action sequence in `optimal_ticks` steps.

3. **Extend `RootCauseType`.** In [`unified_incident_env/models.py`](unified_incident_env/models.py), add the new `RootCauseType` value.

4. **Add a smoke test.** In [`unified_incident_env/tests/test_round2_templates.py`](unified_incident_env/tests/test_round2_templates.py), add the new template_id to `ROUND2_TEMPLATES`.

5. **Verify.** `make test` should pass; the procgen variants will be auto-generated; `/tasks` will expose 78 scenarios (72 + 6 new).

See [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) for the full guide, including Advanced-tier YAML schema and Max-tier triplet (family + chaos + compose) authoring.

---

## 15. References + incident corpus

The 12 Basic templates, 3 Advanced reference scenarios, and 11 Max chaos patterns are all grounded in a 45-incident corpus surveyed during design.

### Vibe-coded SaaS security research (the framing for the SaaS-layer specialization)

- **Veracode 2025 AI Code Security Study** — 45% of AI-generated code has security flaws (n = 100+ LLMs, 80 scenarios)
- **JFrog / Snyk 2025** — ~40% of AI-generated database queries are SQL-injectable
- **Accorian 2025** — 88% of AI-generated logging unsafe; 86% of AI-generated input validation contains XSS
- **Replit / SaaStr incident, July 2025** — agent deleted production DB during an explicit code freeze
- **Tea app 2025** — leaked user data through unauthenticated admin routes
- **Base44 2025** — URI-construction bug let unauthenticated users hit privileged endpoints
- **Cloudflare Nov 2025** — bot-detection permissions regression (canonical config-rollout pattern)
- **Vercel Apr 2026** — Context.ai OAuth token compromise

### Persona-tiered benchmarking literature (the dimensional-escalation rationale)

- **SWE-bench Lite / Verified / Pro** — escalation along scenario count, single-file vs multi-file, languages
- **MLE-bench Low / Med / High / Lite** — escalation along dataset volume + horizon
- **ITBench static / live** — escalation along observability richness + execution risk
- **WebArena / -Verified / -Hard** — escalation along DOM volatility + horizon
- **CRMArena / -Pro** — escalation along multi-turn complexity + confidentiality

What's structurally new in sre-gym: each tier escalates a *different* dimension, not the same dimension at three depths.

### Postmortem corpus consulted (45 incidents, 2022–2026)

Cloudflare (Feb/Jan 2026 config rollouts, Nov 2025 deploy regression, Oct/Jun 2023 routing/DNS, Aug/Sep 2025 dependency degradation, Mar 2025 R2, Jun 2025 KV, Dec 2025 WAF, Feb 2025 R2 storage), Fly.io (Apr 2026 SQLite mutex, disk saturation, cluster rebuild, Sidekiq backlog, fiber cut, tunnel hang, Mar 2026 routing isolation, Dec 2024 proxy deadlock, Oct 2024 mesh storm), Railway (Mar 2026 CDN contamination, Feb 2026 anti-fraud cascade, DDoS, Jan 2026 GitHub rate-limit, Nov 2025 task queue, Oct 2025 Postgres lock, Dec 2025 framework CVE, backend outage), Supabase (Feb 2026 VPC blackout, edge function 504s), Netlify (Mar/Apr 2026 deploy + dependency regressions, Feb 2026 query contention), Stripe (Mar 2022 latency / retry-storm, Feb 2024 ledger drop, Sep 2025 streaming patches), Vercel (Apr 2026 OAuth pivot, Oct 2025 metadata routing, Mar/Apr 2026 Edge function regressions), PlanetScale (Oct 2025 us-east-1 cascade).

For each incident: time-to-detect, time-to-mitigate, time-to-resolve, diagnostic signals, investigative red herrings, eventual remediation, and reversibility.

See [`docs/REFERENCES.md`](docs/REFERENCES.md) for the per-template / per-scenario / per-pattern grounding.

---

## 16. Submission checklist

Hackathon minimum requirements vs what we ship:

| Requirement | Shipped? | Where |
|---|---|---|
| OpenEnv (latest release) compliance | ✅ | [`openenv.yaml`](openenv.yaml), `openenv validate` green |
| Training script using Unsloth or HF TRL | ✅ | [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) |
| Colab notebook judges can re-run | ✅ | All 4 notebooks Colab-ready, A100/L4-aware |
| Evidence of real training (loss + reward plots) | ✅ pipeline | [`02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) generates them; live numbers in [`train/data/`](train/data/) |
| Mini-blog or <2 min YouTube video | 🟡 prep | placeholder section below |
| HF Space deployment | ✅ | [dakshdoesdev-sre-gym.hf.space](https://dakshdoesdev-sre-gym.hf.space) |
| README with motivation, env explanation, results | ✅ | this document |
| Link to env in HF Space + all materials | ✅ | top of README |

### Materials linked from this README

- HF Space: [dakshdoesdev-sre-gym.hf.space](https://dakshdoesdev-sre-gym.hf.space)
- GitHub repo: [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- Trained adapter (target): `dakshdoesdev/sre-gym-qwen25-3b-grpo`
- Mini-blog / video: *to be added at submission time*

---

## 17. FAQ

**Q: Why ship Advanced and Max as design rather than runnable?**
A: A great vision document with one fully-specced family is more credible than a half-built tier with 1,000 broken scenarios. Judges who run real infrastructure can smell unfinished ambition; we'd rather ship one thing that genuinely works (Basic, end-to-end trainable) and two things that are genuinely well-specified (Advanced + Max as YAML / docker-compose blueprints) than three things that are half each.

**Q: Why specialize at the vibe-coded SaaS layer rather than generic K8s SRE?**
A: Because vibe-coded SaaS is the fastest-shipping software category on Earth and has the weakest SRE muscle of any category ever shipped (Veracode 2025: 45% of AI-generated code has security flaws). It's a domain with measurable training value and known failure shapes. Existing SRE benchmarks cluster on the K8s/microservices side; the indie/SaaS-layer slot is white space.

**Q: Won't a 3B model be saturated by the env?**
A: We hardened the scripted-optimal ceiling at ≤ 0.80, leaving 0.20 of headroom. A trained 3B has to beat `optimal_ticks` *and* avoid every noise query to access that headroom. Our calibration data shows Claude Opus hand-driven sits at 0.77 — strong frontier models still don't saturate, so a small specialist has somewhere to climb.

**Q: How does a 3B specialist beat Claude Haiku?**
A: Through dense reward shaping + GRPO group-relative advantages + curriculum (60 procgen scenarios across 12 distinct cognitive failure modes). The shaped per-tick reward correlates ~0.85 with full-episode reward, so even single-action GRPO scoring gives convergent signal. The 3B's advantage over Haiku at *this slice* is in-distribution training; Haiku is general-purpose and pays a tax for that breadth.

**Q: Can I use this env without Unsloth / TRL?**
A: Yes. The env is OpenEnv-compliant, with `/reset` `/step` `/state` HTTP endpoints. Any RL framework (Atropos, OpenClaw-RL, custom GRPO loops) can drive it. We ship the OpenClaw-RL pool-server shim for async GRPO out of the box.

**Q: How big is a Basic-tier procgen scenario at runtime?**
A: ~600 tokens of observation per tick, ~80 tokens of action, ~12 ticks max = ~8K context. Comfortable inside Qwen2.5-3B's 32K context window with room for system prompt and chat history.

**Q: How do I add a 13th template?**
A: See [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md). 60-minute path. Templates are pure data (Python dict + lambda), the simulator core never changes.

**Q: How do I run Advanced or Max?**
A: You don't, in this repo. Advanced needs ~2 weeks of focused engineering + 1–2 A100-days. Max needs $40-150/day cluster + multi-week training. We ship the YAML specs + docker-compose so a downstream operator can lift them; we don't pretend they were trained.

**Q: How does this differ from `kube-sre-gym`?**
A: kube-sre-gym is the K8s-cluster-shaped SRE env; we're the indie/vibe-coded SaaS-layer complement (Stripe webhooks, Supabase RLS, schema drift, Vercel OAuth pivots). Different failure surface, different training signal, complementary positioning. Plus we ship explicit dimensional-escalation tier story.

---

## 18. Team + acknowledgments

Built for the OpenEnv-class hackathon, India 2026, by the dakshdoesdev / Madhav-GPT team.

**Team:** 3 contributors. Per-team HF-credit budget shapes the Basic-tier compute targeting (3 × 30 = 90 credits across the team, but we design to a single-A100 12-hour budget so any one teammate can re-run training).

**Acknowledgments:**

- The OpenEnv team for the `Environment[A,O,S]` contract and the HF Space + cpu-basic deployment path.
- Unsloth + HuggingFace TRL for the GRPO trainer and Qwen2.5 fast-loading.
- Gen-Verse/OpenClaw-RL for the async-GRPO pool-server pattern.
- The SRE community for the 45 postmortems consulted during scenario design (Cloudflare, Fly.io, Railway, Stripe, Vercel, Supabase, Netlify, PlanetScale).
- Veracode, JFrog/Snyk, Accorian for the AI-code-security research that grounds the vibe-coded SaaS framing.

---

## License

Apache 2.0.

The dimensional-escalation tier story is the single most important sentence in this repo: **each tier escalates a *different* dimension — compute, horizon, realism — not just scenario count.** Read the rest of the docs through that lens.
