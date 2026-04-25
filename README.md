---
title: SRE Gym
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# sre-gym — a tier-escalating SRE training environment

**Each tier escalates a different dimension: Basic escalates compute, Advanced escalates horizon, Max escalates realism.** That single sentence is the load-bearing claim of the project. The rest of this README tells you exactly what's in the box, exactly what isn't, and how to verify both.

This README has been rewritten to be honest about state. An earlier draft over-sold what's runnable end-to-end. If you're reading this with the [v3 marketing draft](https://github.com/dakshdoesdev/sre-enginnerllm/blob/0a048ce/README.md) in another tab, prefer this one — it agrees with `openenv.yaml`, `pyproject.toml`, and the actual code.

- **Live HF Space:** [huggingface.co/spaces/Madhav189/sre-env](https://huggingface.co/spaces/Madhav189/sre-env)
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm) — the repo name is misspelled (`enginner`); we kept it because rotating the URL after submission is worse than living with the typo.
- **OpenEnv manifest:** [`openenv.yaml`](openenv.yaml) — single source of truth for which tier is runnable in which mode. Differences between this README and the manifest are bugs in the README; the manifest wins.
- **Tests:** 203 collected via `pytest --collect-only -q`.

---

## What's actually runnable today (v3.1, honest)

| Tier | Runnable kind | Scenarios | What "running" means | Trained model? |
|---|---|---|---|---|
| **Basic** | ✅ live HTTP env | 12 base templates × 6 entries each (1 base + 5 procgen) = **72 scenarios** | `/reset` + `/step` against the FastAPI server in this Docker image. The Gradio UI drives a real episode end-to-end via these routes. | ❌ pending — `notebooks/01_basic_train_grpo_unsloth.ipynb` is shipped but **never executed** in this repo |
| **Advanced** | 🟡 Python orchestrator | 3 reference YAML scenarios | `sre_gym.advanced.runner.run_advanced` chains Basic episodes together, threading horizon state (unresolved alerts, pending deploys, tech-debt counter, horizon-decay reward). The 28-action universe declared in the YAML is **design spec only**; the runner uses the Basic 11 actions. | n/a |
| **Max** | 🟡 Python state-machine sim | 1 family × 12 chaos patterns (one is the alias `payment_webhook_storm`) | `sre_gym.max.runner.run_max` mutates an in-memory 22-node graph using the Basic 11-action interface. The docker-compose stack under `sre_gym/max/families/.../compose/` references `ghcr.io/sre-gym/*` images that are **not published** — treat that compose file as design spec. | n/a |

**Important honesty caveats** the prior README under-stated:

- **Training has not been run end-to-end in this repo.** All 6 notebooks have 0 executed cells, 0 outputs, and there are 0 PNG/JPG plots committed. The notebooks are real Colab-runnable scripts; running them is on the operator (us, after submission window allows it). The README baseline tables show frontier-LLM measurements only.
- **Of 12 Basic templates, 6 have zero teacher trajectories committed.** The 6 round-2 templates (`auth_token_expiry`, `dep_degradation`, `memory_leak_oom`, `migration_lock`, `network_partition`, `rate_limit_retry_storm`) were added without an accompanying SFT trajectory collection pass. Plan: collect them via `train/collect_trajectories.py` against Claude Opus / Llama-3.3-70B before kicking off GRPO.
- **`seed_combined.jsonl` has 21 rows, not the ≥200 the original execution.md checklist required.** That checklist was aspirational. We're updating it to reality in this commit.
- **Advanced YAMLs declare 42 actions across the 3 scenarios that have no Python implementation** (`feature_flag_toggle`, `escalate_security`, `query_session_cardinality`, `propose_postmortem`, etc.). Each YAML now carries a prominent `DESIGN-SPEC HEADER` listing the implemented 11 actions explicitly, so future readers can't miss the gap.
- **The Supabase-RLS Advanced scenario maps to non-Supabase Basic templates** under the hood (`payment_webhook_misconfig` / `migration_lock` / `worker_deploy_cascade`). The narrative wrapper is real; the underlying simulation is approximate.
- **All 12 Max chaos `deploy_marker`s carry the same date** (`2026.04.25`). Those are synthetic markers used by `query_deploys()` inside the simulator; they're not real-incident citations and shouldn't be read as such.

These caveats are uncomfortable but they're in the codebase; pretending otherwise was the original mistake.

---

## Quickstart (3 commands)

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm && cd sre-enginnerllm
pip install -e '.[dev]'
uvicorn app:app --host 0.0.0.0 --port 7860     # opens the Gradio terminal at http://localhost:7860
```

Or drive each tier from the CLI:

```bash
make baseline                                                                  # Basic — runs scripted-optimal across all 12 templates
python -m sre_gym.advanced run cascading_release_train --seed 1                # Advanced — chained-Basic-episode trace
python -m sre_gym.max run ecommerce_vibecoded_saas --chaos rls_silent_leak     # Max — graph-mutator trace
```

---

## The Basic tier — what the env actually does

12 base templates, each with 5 procedurally-generated variants (so 6 entries per template, 72 scenarios total). Topology is fixed at 4 services: `api-gateway / cache / database / worker`.

| # | Template | Difficulty | Skill the agent must learn | Has teacher data? |
|---|---|---|---|---|
| 1 | `worker_deploy_cascade` | easy | deploy-history reasoning | ✅ |
| 2 | `db_config_rollout` | medium | config-vs-code disambiguation | ✅ |
| 3 | `gateway_auth_rollout` | hard | wrong-loud-service trap | ✅ |
| 4 | `payment_webhook_misconfig` | medium | downstream symptom (Stripe) | ✅ |
| 5 | `schema_drift_missing_migration` | medium | application-vs-DB blame | ✅ |
| 6 | `cache_stale_state` | medium | metrics-up-but-customers-down | ✅ |
| 7 | `dep_degradation` | medium | "your service vs theirs" | ❌ — no teacher data yet |
| 8 | `memory_leak_oom` | hard | restart count > error count | ❌ — no teacher data yet |
| 9 | `auth_token_expiry` | medium | cross-service credential propagation | ❌ — no teacher data yet |
| 10 | `network_partition` | hard | trust connectivity, not self-reports | ❌ — no teacher data yet |
| 11 | `rate_limit_retry_storm` | hard | counterintuitive (more retries = worse) | ❌ — no teacher data yet |
| 12 | `migration_lock` | medium | lock contention without crash | ❌ — no teacher data yet |

**Action space (11 actions, validated by Pydantic Literal — see `unified_incident_env/models.py`):** `query_logs / query_metrics / query_dependencies / query_deploys / rollback_deploy / restart_service / isolate_service / run_check / submit_hypothesis / escalate / declare_resolved`.

**7-dimension grader:** recovery (0.25) + containment (0.15) + verification (0.20) + impact (0.05) + efficiency (0.05) + speed_bonus (0.0–0.10) + noise_handling (0.0–0.05). Hardened scripted-optimal ceiling **≤ 0.80** leaves 0.20 of headroom for a trained agent. The CI invariant `test_baseline_ceiling_is_hardened_below_080` enforces both edges of `[0.70, 0.80]` across all 12 templates.

Full reward design with the potential-shaping derivation: [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md).

---

## The Advanced tier — chained Basic episodes (be honest about what this is)

Three reference scenarios shipped, all runnable via [`sre_gym/advanced/runner.py`](sre_gym/advanced/runner.py):

| Scenario | What it tests | Phases | Backing Basic templates |
|---|---|---|---|
| [`cascading_release_train`](sre_gym/advanced/scenarios/cascading_release_train.yaml) | long-horizon state tracking, recovery from early mistakes | 2 | `schema_drift_missing_migration`, `dep_degradation` |
| [`observability_pipeline_outage`](sre_gym/advanced/scenarios/observability_pipeline_outage.yaml) | partial observability, alternate-path investigation | 2 | `cache_stale_state`, `worker_deploy_cascade` |
| [`supabase_rls_silent_leak`](sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml) | security-aware response, containment-first discipline | 3 | `payment_webhook_misconfig`, `migration_lock`, `worker_deploy_cascade` |

**The honest framing:** the runner chains those Basic templates together with persistent horizon state — unresolved alerts ride into the next phase, deploys rolled back without a follow-up restart get a deferred-restart tax, the tech-debt counter scales subsequent step costs, and the final reward is `mean(per-phase rewards) × 0.92^unresolved_phases`. That mechanism is implemented end-to-end and produces deterministic, comparable trajectories.

**What it isn't:** a faithful 15–20 service simulator. The YAML topologies declare 18+ services and 28-ish actions per scenario; only 4 services and 11 actions are implemented in the env. The wider universe is a **design spec** documented in the YAML so a downstream operator with 1–2 A100-days could implement it. Each Advanced YAML now carries a `DESIGN-SPEC HEADER` calling that out explicitly.

Full design defence: [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md).

---

## The Max tier — Python state-machine simulator (no real cluster)

One specced family (`ecommerce_vibecoded_saas`, 22 services in YAML) plus a 12-pattern chaos library (one is the alias `payment_webhook_storm`).

The **runnable** Max surface is the in-memory state machine at [`sre_gym/max/runner.py`](sre_gym/max/runner.py): same 11 actions as Basic, applied as transitions over a graph of `ServiceNode` objects. Reward is the potential-shaped function over graph health.

| Pattern | Correct action | Implementation status |
|---|---|---|
| `deploy_regression` | `rollback_deploy(orders-service)` | ✅ runnable in Python sim |
| `stripe_webhook_signature_regression` | `rollback_deploy(api-gateway)` | ✅ runnable |
| `dependency_degradation` | `rollback_deploy(redis-sessions)` | ✅ runnable |
| `config_rollout` | `rollback_deploy(api-gateway)` | ✅ runnable |
| `retry_storm` | `rollback_deploy(worker-payments)` | ✅ runnable |
| `migration_lock` | `rollback_deploy(postgres-primary)` | ✅ runnable |
| `rls_silent_leak` | `rollback_deploy(postgres-primary)` | ✅ runnable (security-classified) |
| `oauth_supply_chain_pivot` | `isolate_service(vercel-frontend)` | ✅ runnable (security-classified, only `isolate_service` pattern) |
| `observability_self_denial` | `rollback_deploy(orders-service)` | ✅ runnable |
| `secondary_rate_limit` | `rollback_deploy(worker-orders)` | ✅ runnable |
| `cdn_cache_contamination` | `rollback_deploy(vercel-edge-fn)` | ✅ runnable (security-classified) |
| `payment_webhook_storm` (alias) | `rollback_deploy(api-gateway)` | ✅ runnable |

**Action distribution: 11 of 12 patterns reduce to `rollback_deploy(<service named verbatim in incident_summary>)`.** That's a known limitation — the simulator is more useful as a fault-injection harness than as a hidden-information puzzle. We document it openly rather than burying it.

**The compose file under [`sre_gym/max/families/.../compose/ecommerce.yaml`](sre_gym/max/compose/ecommerce.yaml) is design-spec only.** Every `image:` line points at `ghcr.io/sre-gym/<service>:1.0`; **none of those images are published**. `docker compose up` will fail at the first `docker pull`. The compose file documents the topology shape an operator could provision; it does not run.

Full design + operator notes: [`docs/MAX_TIER.md`](docs/MAX_TIER.md).

---

## The HF Space UI

The Space serves a single Gradio page at `/`. Every existing FastAPI route — `/reset`, `/step`, `/state`, `/tasks`, `/baseline`, `/grader`, `/status`, `/health`, `/metadata`, `/schema`, `/info`, `/simple`, `/docs`, `/redoc`, `/openapi.json`, `/mcp`, `/mcp/tools`, `/mcp/reset` — stays accessible on the same port (7860).

The UI has one input row (tier radio + HF token + provider + model + provider key), one streaming terminal pane, and one ▶ RUN EVAL button that loops the held-out set per tier:

| Tier | Held-out set the UI loops | Source of truth |
|---|---|---|
| Basic | 12 `__p05` procgen variants | [`eval/holdout_basic.json`](eval/holdout_basic.json) |
| Advanced | 3 reference scenarios | `sre_gym/advanced/scenarios/*.yaml` |
| Max | 12 chaos patterns | `CHAOS_PATTERNS` in [`sre_gym/max/runner.py`](sre_gym/max/runner.py) |

Tokens live only in `gr.State` for the browser session. They are never written to disk, never logged, never echoed in the terminal pane.

---

## Training pipeline (Basic only) — not yet executed

The pipeline lives in [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) and [`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb). It targets Qwen2.5-3B in 4-bit via Unsloth, LoRA r=64, SFT cold-start (500 steps) → GRPO (800 steps), ~12h on a single A100, ~$30 of HF credits.

Notebook 02 runs the comparison sweep and writes:

- `eval/results/comparison_raw.csv`
- `eval/results/comparison_summary.csv`
- `eval/results/comparison_table.csv`
- `eval/results/comparison_per_template.png`
- `eval/results/comparison_hero.png`

**None of these artifacts exist in the repo today.** The notebooks need a real run. We will execute them on Colab and commit the resulting plots + CSV to `eval/results/` post-submission-window.

---

## Frontier baselines (Basic, raw measurements only — no trained model row)

The earlier README had a `Trained Qwen2.5-3B (target — pending GRPO run)` row in this table. That row was misleading because it implied a measured number was forthcoming inside this commit. It's been removed until a real run produces a real number.

What is actually measured today, sourced from `train/data/`:

| Policy | Episodes | Resolved | Mean score | Source JSONL |
|---|---|---|---|---|
| Heuristic (deterministic) | 18 | 0/18 | **0.187** | [`eval_sweep_baselines.jsonl`](train/data/eval_sweep_baselines.jsonl) |
| Random (uniform over allowed actions) | 18 | 0/18 | **0.230** | [`eval_sweep_baselines.jsonl`](train/data/eval_sweep_baselines.jsonl) |
| Llama-3.3-70B-Versatile (Groq) | 11 | 5/11 | **0.42** | `llama33_70b_groq_*.jsonl` |
| Llama-3.3-70B-Instruct (Fireworks) | 4 | 3/4 | **0.73** | `llama33_70b_smoke4.jsonl` |
| Scripted-optimal baseline | 12 | 12/12 | **≤ 0.80** (CI-enforced) | enforced by `test_baseline_ceiling_is_hardened_below_080` |
| Claude Opus 4.7 (hand-driven) | 6 | 6/6 | **0.77** | `claude_seed.jsonl` |

Two honest caveats on this table:

1. **Sample sizes are uneven (4 / 6 / 11 / 18).** That's because they were collected as smoke tests, not as a controlled benchmark. A judge reading them as a benchmark would be right to discount the 4-episode Llama row in particular.
2. **Random outperforms our hand-written heuristic (0.230 vs 0.187).** That's not flattering. The heuristic is a 7-line if-else; the env's shaped reward gives small positive credit even to "wrong" actions like `query_logs` because they consume a step but reveal evidence. Random sometimes stumbles into a useful evidence-gathering sequence; the heuristic always commits to a fixed (often wrong) sequence. The fix is in the heuristic, not the env, and is on the v3.1 list.

When a real GRPO run produces a `Trained Qwen2.5-3B` row, it'll land here with `n=36` (12 held-out × 3 seeds) and a `comparison_hero.png` link.

---

## Tier-aware Python API

```python
from sre_gym import SREGym, Tier

# Basic — per-step (live FastAPI) or end-to-end
env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
result = env.run("memory_leak_oom__p02", seed=42)

# Advanced — episodic only (chained Basic episodes)
env = SREGym(tier=Tier.ADVANCED)
result = env.run("cascading_release_train", seed=1)
print(result.summary())

# Max — per-step (graph mutations) or end-to-end (state-machine simulator)
env = SREGym(tier=Tier.MAX)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

---

## Tests + lint

```bash
make test            # 203 collected, all green at HEAD
ruff check .          # clean
openenv validate .    # green (uv.lock generated)
```

Test count history (the prior README claimed 140 — that number was stale; the manifest didn't have the parametrize expansions):

| Source | Reported number | Actual today |
|---|---|---|
| README earlier draft | 140 | — |
| `execution.md` earlier draft | 74 | — |
| `demo/pitch.md` earlier draft | 21 | — |
| `pytest --collect-only -q` | — | **203** |

The demo/pitch.md line was a v1 artifact ("21 tests passing" was correct *for v1*); we've left an updated pitch.md in the repo.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  app.py  (uvicorn app:app on port 7860)                      │
│   ├─ Gradio terminal UI mounted at /                         │
│   └─ FastAPI server (unified_incident_env.server.app)        │
│       ├─ /reset /step /state         OpenEnv contract        │
│       ├─ /tasks /baseline /grader    catalogue + scoring     │
│       ├─ /status /health             ops probes              │
│       ├─ /metadata /schema           OpenEnv metadata        │
│       ├─ /mcp /mcp/tools /mcp/reset  JSON-RPC 2.0 dual-route │
│       ├─ /docs /redoc /openapi.json  Swagger / ReDoc         │
│       └─ /info /simple               legacy markdown landing │
│                                                              │
│  sre_gym/                                                    │
│   ├─ tier.py            Tier enum + TierConfig               │
│   ├─ env.py             SREGym factory (delegates per tier)  │
│   ├─ basic_runner.py    wrap UnifiedIncidentEnvironment      │
│   ├─ advanced/runner.py chain Basic episodes + horizon state │
│   ├─ max/runner.py      Python state-machine over 22 nodes   │
│   ├─ ui/                providers, router, policies, runner  │
│   └─ exceptions.py      typed errors                         │
└──────────────────────────────────────────────────────────────┘
```

Full architectural narrative — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Per-tier deep dives in [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md) / [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md) / [`docs/MAX_TIER.md`](docs/MAX_TIER.md). Operator guide: [`execution.md`](execution.md).

---

## Honest known issues (the things the earlier README didn't say)

We're listing these explicitly so a reviewer doesn't have to dig:

- **Training is pending.** Notebooks ship; a real GRPO run does not.
- **6 of 12 templates have no teacher trajectories** (`auth_token_expiry`, `dep_degradation`, `memory_leak_oom`, `migration_lock`, `network_partition`, `rate_limit_retry_storm`).
- **`seed_combined.jsonl` has 21 rows.** SFT cold-start would need ≥ 200 for stability.
- **42 actions in Advanced YAMLs have no Python implementation.** Each YAML now carries a `DESIGN-SPEC HEADER` listing the implemented 11 explicitly.
- **Max chaos patterns reduce to ~2 actions.** 11 of 12 are `rollback_deploy(<service-named-in-summary>)`. The simulator is a fault-injection harness, not a hidden-information puzzle.
- **Max compose images aren't published.** `docker compose up` will fail.
- **Random > Heuristic** in the published baseline data. Real and embarrassing; fix is on the heuristic side.

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, India 2026 — by the dakshdoesdev / Madhav189 team.
