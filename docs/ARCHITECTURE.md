# Architecture

> **The single insight that ties the whole pitch together: each tier escalates a *different* dimension — compute (Basic) → horizon (Advanced) → realism (Max) — not just scenario count.** If you only read one paragraph of this document, read this one.

This document explains why sre-gym ships three tiers, what each tier defends as a research question, and how the package is structured to make that defence visible from a 30-second skim of the repo.

---

## 1. The dimensional-escalation insight

The standard pattern in agentic-LLM evaluation, surveyed across SWE-bench Lite/Verified/Pro, MLE-bench Low/Med/High, ITBench-static/live, WebArena/-Verified/-Hard, and CRMArena/-Pro, is to escalate a single axis (volume, complexity, dataset size, horizon length) across difficulty bands. That works for benchmarks whose underlying capability is one-dimensional. SRE is not one-dimensional.

A junior on-call engineer learning to triage faces a fundamentally different bottleneck (cognitive efficiency under tight context) than a senior SRE running a multi-incident postmortem (state tracking across long horizons), which faces a fundamentally different bottleneck than an enterprise platform team operating against an actively chaos-engineered cluster (operating in a partially-observable, adversarial, irreversible world). Their training signals, episode shapes, observation richness, and reward structures should not look the same.

sre-gym takes that observation seriously and stratifies its tiers along *the dimension that actually limits the persona's training loop*:

| Tier | Bottleneck | Persona implication | Episode shape |
|---|---|---|---|
| Basic | **Compute** ($30 of HF credits, 1 A100 ~12h) | Pre-digested observations, dense reward shaping, 8K context, 11-action space | 8–13 ticks |
| Advanced | **Horizon** ($300–500 budget, 1–2 A100-days) | Multi-incident chains, partial observability, 28-action space, on-call peer | 60–90 ticks |
| Max | **Realism** (8×A100/H100, real chaos eng) | Ephemeral docker-compose / k3d, real `kubectl` / Vercel / Stripe APIs, subprocess shell, learned-critic rewards | 110–180+ actions, real wall-clock minutes |

This framing is *defensible as research*: the claim is that "training on a tier-1 environment that's causally rich but compute-cheap will produce a 3B specialist that beats Haiku on incident triage" is testable in 12 hours. The claim that "training on a tier-3 environment that includes real subprocess-shell access will produce an SRE agent that can actually go on-call" is testable in *months* and tens of thousands of dollars. Those are different research questions and the environment should make them visibly different.

---

## 2. Repository layout, with the design intent annotated

```
sre-enginnerllm/
├── sre_gym/                          # Tier-aware public package
│   ├── env.py                        # SREGym(tier=...) factory; Basic delegates,
│   │                                 # Advanced/Max raise TierNotRunnableError
│   │                                 # carrying a docs pointer.
│   ├── tier.py                       # Tier enum + TierConfig; the
│   │                                 # escalation_dimension field is the
│   │                                 # load-bearing piece of the pitch.
│   ├── advanced/scenarios/*.yaml     # 3 reference scenarios, real YAML, real
│   │                                 # topology, real reward dimensions, real
│   │                                 # reference traces — but the simulator
│   │                                 # backing them is intentionally not built.
│   └── max/                          # 1 fully-specced family, real
│       ├── families/*.yaml           # docker-compose, real chaos library,
│       ├── chaos/*.yaml              # real workload generator config.
│       └── compose/*.yaml            # Provisioning is left to the operator.
│
└── unified_incident_env/             # Basic-tier core; the v2 surface that the
                                      # HF Space serves and openenv.yaml declares.
```

Two design choices worth calling out:

**(a) Basic delegates to `unified_incident_env` rather than reimplementing.** The Basic tier's runnable surface is the existing v2 environment, kept verbatim — including its 36+ test suite, scripted-optimal baselines, and HF Space deployment. The `sre_gym` wrapper is intentionally thin: it adds the tier flag, the introspection methods, and the YAML-spec loader for the design-only tiers. This is the difference between "a single-tier env that's hard to extend" and "an env that visibly carries the three-tier story while still passing every Basic-tier test it ever passed."

**(b) Advanced and Max are shipped as data, not as code.** The YAML specs in `sre_gym/advanced/scenarios/` and `sre_gym/max/families/` are *real*: they reference real topologies, real action sets, real reward dimensions, real chaos patterns, and real reference traces. They're loaded by `SREGym.list_scenarios()` and renderable by the playground. What's missing is the simulator backing them — and that's deliberate. Building a credible Max-tier simulator in 36 hours is a fantasy; building a credible Max-tier *spec* that downstream operators can lift into a $40-150/day cluster is achievable. The cost of pretending Max is shipped runnable would be zero credibility with judges who have run real chaos engineering. The cost of shipping it as a credible vision is one extra YAML file.

---

## 3. The Basic tier in detail

The Basic tier is the only tier we trained against, so it gets the most concrete defence.

### 3.1 What it is

12 base templates × 5 procgen variants = 72 deterministic scenarios. Procgen jitters metric values, deploy timestamps, and noise-service rotation while preserving the causal structure — so a trained agent can't memorize fingerprints. Holding out one variant per template gives a 60-train / 12-eval split.

### 3.2 The 12 templates and what each one teaches

| # | Template | Skill | Decoy / red herring |
|---|---|---|---|
| 1 | `worker_deploy_cascade` | deploy-history reasoning | none — the easy entry point |
| 2 | `db_config_rollout` | config-vs-code disambiguation | concurrent worker deploy |
| 3 | `gateway_auth_rollout` | wrong-loud-service trap | worker queue-depth alert |
| 4 | `payment_webhook_misconfig` | downstream symptom (Stripe) | DB write-rate drop |
| 5 | `schema_drift_missing_migration` | application vs DB blame | DB looks healthy |
| 6 | `cache_stale_state` | metrics-look-good-but-customers-don't | cache hit rate is *up* |
| 7 | `dep_degradation` | "your service vs theirs" | worker CPU is loud |
| 8 | `memory_leak_oom` | restart count > error count | DB CPU spikes look like DB fault |
| 9 | `auth_token_expiry` | cross-service credential propagation | gateway is the loudest service |
| 10 | `network_partition` | trust connectivity, not self-reports | cache reports healthy in own metrics |
| 11 | `rate_limit_retry_storm` | counterintuitive (more retries = worse) | DB CPU/connections look pathological |
| 12 | `migration_lock` | lock contention without crash | worker errors look like a worker fault |

Each template contains a different **cognitive failure mode**. A 12-template catalogue with 12 different failure modes is a denser training signal than a 60-template catalogue that all reduce to "look at the deploy that just happened" — that's the depth-not-quantity argument.

### 3.3 Why these compute knobs

- **8K context** — fits the entire Basic episode (12 ticks × ~600 tokens of observation + ~80 tokens of action) inside the trained model's working set without truncation. Letting the trajectory spill into 16K context would force a smaller model (more A100 hours per token) or a longer training run (more wall-clock).
- **11 actions** — small enough that GRPO group-relative advantages converge in 600–1000 steps. Add 5 more actions and the policy has 50% more dimensions to explore at the same compute budget, which usually means a noisier gradient.
- **5-component dense reward** — recovery + containment + verification + impact + efficiency, with shaped intermediate signal (potential-function differences). Pure terminal rewards converge slower than dense shaping at the same compute budget; that's a well-known result and we're paying it.
- **12 templates × 5 variants procgen** — enough variety that a held-out variant is genuinely held-out, but not so much that scenario-specific overfitting eats the training budget.

These knobs collectively are the "compute" in "compute-bounded". Tighten any of them and a Series-A-class operator would call it out as theatre.

### 3.4 What "compute-bounded" actually means in numbers

A representative training run looks like:

| Phase | Steps | Compute | Wall-clock on A100 40GB | Output |
|---|---|---|---|---|
| Seed dataset build | 200 (Claude-driven) | ~$15 of API spend | 2h | `train/data/seed_combined.jsonl` |
| SFT cold start | 500 steps, batch 4 | ~3h | 3h | LoRA r=64 adapter |
| GRPO online | 800 steps, K=4 rollouts | ~6h | 6h | trained adapter |
| Eval sweep | 36 episodes (3 per template) | ~30min | 30min | `eval/results/comparison.csv` |
| **Total** | | **~$15 API + 12h GPU** | **12h** | trained 3B + comparison table |

That's the $30-of-HF-credits budget the design targets, comfortably.

---

## 4. The Advanced tier in detail

### 4.1 The horizon escalation, formalized

Advanced is bounded by horizon: episodes are 60–90 ticks instead of 12, multi-incident chains span 5+ minutes of simulated time, and the agent must track state that no single 8K context window can hold. Three properties make this a different research question:

1. **Multi-incident composition.** One template's resolution can become another template's setup state. Scenario 1 (`cascading_release_train.yaml`) is the canonical example: rolling back the gateway is correct, but it triggers a downstream worker drift that materializes 25 ticks later. The agent has to recognize the *chained* incident as caused by their own fix and reach for a second rollback rather than treating it as a fresh outage.
2. **Partial-observability noise.** Sometimes `query_logs` returns degraded data because the logging pipeline is the affected service. Scenario 2 (`observability_pipeline_outage.yaml`) is the canonical example: the agent must drop log sampling and toggle verbose-logging off *before* attempting root-cause diagnosis, because the diagnostic tool is itself broken.
3. **Cross-domain reasoning.** Scenario 3 (`supabase_rls_silent_leak.yaml`) is a reliability incident with a security root cause. The agent must classify it correctly (reach for `escalate_security` rather than the platform on-call), contain the data leak via feature flag *before* rolling back, and produce a postmortem with a leak-window calculation. No existing SRE benchmark scores cross-domain reasoning.

### 4.2 Why this isn't trained in this repo

A faithful Advanced simulator would need:
- a 15–20 service event-loop simulator (not a 4-service one)
- multi-tick fault propagation (one fix triggering a chained fault N ticks later, with proper causal latency)
- a synthetic on-call-peer model that responds to escalations
- ~28 action handlers, vs. 11 in Basic
- a learned-critic reward path for postmortem quality

That's roughly 2 weeks of focused engineering and 1–2 A100-days of training. Both are out of scope for the 36-hour hackathon window. We ship the design at the YAML level so that a downstream operator with the budget can lift it; we do not pretend it was trained.

---

## 5. The Max tier in detail

### 5.1 The realism escalation, formalized

Max is bounded by realism. The world stops being a simulator. A `reset()` provisions a fresh 22-service docker-compose stack (Vercel + Supabase + Stripe + Postgres + Redis + Kafka + 3 worker pools + observability stubs + chaos controller). The agent's `rollback_deploy` is a real `kubectl rollout undo` against that stack. `query_logs` reads from a real Loki/Promtail pipeline. `query_traces` reads from a real Tempo cluster. Faults are injected via a real Chaos-Mesh-style chaos library. Reward is computed from the actual recovery state of the actual stack.

Three properties are unique to Max:

1. **Real subprocess access.** The agent has a sandboxed shell. It can write code, commit it to a sandboxed git mirror, push, watch CI, observe a deploy, roll back. This is the "real hard work instead of exploiting shortcuts" spec from Theme #3.1 of the OpenEnv brief.
2. **Real action irreversibility.** A real Stripe refund is a real Stripe refund. The cluster is destroyed on next reset, but actions taken at the application layer are real.
3. **Outcome-scored rewards.** A second small model evaluates the agent's postmortem against the actual recovery trajectory. Reward is no longer a deterministic rubric — it's an outcome judgement, with shaping signals as auxiliary.

### 5.2 Why one family, not 30

Per the design rationale: one fully-specced family (e-commerce + Stripe + Supabase + Vercel) with `compose.max.yaml`, an 11-pattern chaos library, a workload generator config, a reference instance with an expected 110-action trajectory, and operator notes for cost / isolation / safety is more credible than a vague "30+ scenario families" claim. Judges who run real infrastructure would ask "show me one"; if you can't, the whole tier framing collapses. The `ecommerce_vibecoded_saas` family is the show-me-one.

### 5.3 What's deliberately not in this repo

- The published stub images (`ghcr.io/sre-gym/*`) — publishing them is a $1–2k registry-cost commitment that doesn't fit the hackathon-window budget.
- A running cluster — bringing the Max tier up costs $40–150/day depending on cluster size and chaos cadence.
- A trained model against Max — that's a multi-week, multi-A100 commitment.

What *is* in this repo is the spec at the level of detail a downstream operator can actually act on: docker-compose, chaos-library YAML, workload generator config, family-level scenario population spec, and operator-notes block covering cost, isolation, and reset safety.

---

## 6. The contract that ties Basic, Advanced, and Max together

All three tiers share the same five abstract objects:

| Object | Basic concrete | Advanced concrete | Max concrete |
|---|---|---|---|
| Topology | 4 services hard-coded | 15–20 services in YAML | 22 services in docker-compose |
| Action set | 11, Pydantic-validated | 28, validated against `allowed_actions:` list | 50+, including subprocess shell |
| Observation | pre-digested fields | noisy multi-source feed | raw Prometheus / Loki / Tempo |
| Reward | 7-dim deterministic rubric | rubric + chained-incident bonus + postmortem critic | outcome + learned-critic + IaC-remediation bonus |
| Episode | 8–13 ticks | 60–90 ticks | unbounded (real wall-clock minutes) |

Same shape, escalating depth. That's what makes the tier story coherent rather than three unrelated environments stacked in one repo.

---

## 7. OpenEnv framework integration

Basic uses:
- `openenv.core.env_server.Environment[A, O, S]` base class
- Typed Pydantic `Action / Observation / State`
- `/reset` `/step` `/state` HTTP endpoints via `create_fastapi_app`
- `max_concurrent_envs` for batched rollouts (the GRPO contract)
- Custom `/tasks` `/baseline` `/grader` `/status` `/health` extension routes for scenario-catalog-and-grader introspection

Advanced and Max would extend this with:
- `MCPEnvironment` base + `@self.tool()`-registered actions for production serving
- WebSocket `/ws` transport for low-latency multi-agent rollouts
- `ServerMode.SIMULATION` vs `ServerMode.PRODUCTION` switch for tier-3 real-API actions
- Custom Gradio `TabbedInterface` with a "topology inspector" tab

These are documented in the per-tier docs but not implemented here.

---

## 8. The judging-criteria mapping

The OpenEnv hackathon's published rubric weights are: Innovation 40, Storytelling 30, Reward Curves 20, Reward/Pipeline 10. sre-gym is built so each weight has a concrete corresponding artifact:

| Weight | Artifact in this repo |
|---|---|
| Innovation 40% | The dimensional-escalation tier story, defensible across compute / horizon / realism axes. |
| Storytelling 30% | This document + `README.md` first paragraph + 12-template skill table + reference traces in YAMLs. |
| Reward curves 20% | `eval/results/` (populated by `02_basic_eval_comparison.ipynb`) + `train/data/eval_sweep_baselines.jsonl` reference numbers. |
| Reward/Pipeline 10% | Composable rubric in `unified_incident_env/server/grader.py` + GRPO loop in `01_basic_train_grpo_unsloth.ipynb`. |

Weight by weight, the artifact you'd hand a judge to defend that score is in this repo.
