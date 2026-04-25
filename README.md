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

**Each tier escalates a different dimension: Basic escalates compute, Advanced escalates horizon, Max escalates realism.** Most SRE benchmarks pretend that "more scenarios" is the only escalation axis. It isn't. A 3B specialist on $30 of HF credits faces a fundamentally different bottleneck (efficiency under tight context) than a Series-A startup running a 7B on $300 of compute (state tracking across long horizons), which faces a fundamentally different bottleneck than an enterprise running a 70B against a real chaos-engineering cluster (operating in a partially-observable, irreversible world).

## What's runnable today

| Tier | Runnable | Scenarios | Recommended model | Bound by |
|---|---|---|---|---|
| **Basic** | ✅ on HF Space | 12 templates × 5 procgen = **72** | trained Qwen2.5-3B / Qwen2.5-7B / Llama-3.1-8B | compute |
| **Advanced** | ✅ on HF Space | 3 chained-incident scenarios | Llama-3.3-70B (BYOK) / Claude Sonnet (BYOK) | horizon |
| **Max** | ✅ on HF Space | 1 family × 11 chaos patterns | Claude Sonnet / Opus (BYOK) / GPT-5 (BYOK) | realism |

Every tier ships an interactive UI on the HF Space. Training charts/curves ship for Basic only (Colab notebook). Advanced runs as **chained Basic episodes with persistent horizon state**; Max runs as a **Python state-machine simulator over the 22-node service graph** (the real docker-compose stack lives in `sre_gym/max/families/...` for users with Docker).

## Quickstart (3 commands)

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm && cd sre-enginnerllm
pip install -e '.[dev]'
python app.py        # launches the Gradio terminal UI on http://localhost:7860
```

Or drive each tier from the CLI:

```bash
make baseline                                                                  # Basic, all 12 templates
python -m sre_gym.advanced run cascading_release_train --seed 1                # Advanced
python -m sre_gym.max run ecommerce_vibecoded_saas --chaos rls_silent_leak     # Max
```

- **HF Space (live):** [huggingface.co/spaces/Madhav189/sre-env](https://huggingface.co/spaces/Madhav189/sre-env)
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- **OpenEnv compliance:** `openenv validate` green; **140 tests passing**.

---

## The Basic tier — what's runnable

12 base templates × 5 procgen variants = 72 deterministic scenarios over a 4-service topology (api-gateway / cache / database / worker). Each template adds a **distinct SRE skill**:

| # | Template | Difficulty | Skill the agent must learn | Incident grounding |
|---|---|---|---|---|
| 1 | `worker_deploy_cascade` | easy | deploy-history reasoning | classic deploy-cascade class |
| 2 | `db_config_rollout` | medium | config-vs-code disambiguation | Cloudflare Nov 2025 |
| 3 | `gateway_auth_rollout` | hard | wrong-loud-service trap | Base44 |
| 4 | `payment_webhook_misconfig` | medium | downstream-symptom (Stripe) | Stripe sig drift |
| 5 | `schema_drift_missing_migration` | medium | application-vs-DB blame | Prisma/Supabase |
| 6 | `cache_stale_state` | medium | metrics-up-but-customers-down | session-leak class |
| 7 | `dep_degradation` | medium | "your service vs theirs" | Cloudflare R2 Mar 2025 |
| 8 | `memory_leak_oom` | hard | restart count > error count | OOM-loop class |
| 9 | `auth_token_expiry` | medium | cross-service credentials | Vercel Apr 2026 |
| 10 | `network_partition` | hard | trust connectivity, not self-reports | Fly.io Apr 2026 |
| 11 | `rate_limit_retry_storm` | hard | more retries = worse | Stripe Mar 2022 |
| 12 | `migration_lock` | medium | lock contention without crash | Railway Oct 2025 |

11-action interface: query_logs / query_metrics / query_dependencies / query_deploys / rollback_deploy / restart_service / isolate_service / run_check / submit_hypothesis / escalate / declare_resolved.

7-dimension grader: recovery (0.25) + containment (0.15) + verification (0.20) + impact (0.05) + efficiency (0.05) + speed_bonus (0.0–0.10) + noise_handling (0.0–0.05). Plus per-tick potential-shaped reward. Hardened scripted-optimal ceiling **≤ 0.80** leaves 0.20 of headroom for a trained agent.

Full design: [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md) and [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md).

---

## The Advanced tier — chained Basic episodes

Three reference scenarios shipped, all runnable via [`sre_gym.advanced.runner`](sre_gym/advanced/runner.py):

| Scenario | Tests | Phases |
|---|---|---|
| [`cascading_release_train`](sre_gym/advanced/scenarios/cascading_release_train.yaml) | long-horizon state tracking, recovery from early mistakes | 2 |
| [`observability_pipeline_outage`](sre_gym/advanced/scenarios/observability_pipeline_outage.yaml) | partial observability, alternate-path investigation | 2 |
| [`supabase_rls_silent_leak`](sre_gym/advanced/scenarios/supabase_rls_silent_leak.yaml) | security-aware response, containment-first discipline | 3 |

The runner chains Basic episodes with **persistent horizon state**:
- Unresolved alerts ride into the next phase as additional noise.
- Pending deploys (rolled back without restart) carry a deferred-restart tax.
- Tech-debt counter accumulates from no-progress ticks; scales subsequent step costs.
- Final reward = `mean(per-phase rewards) × 0.92^unresolved_phases`.

Full design: [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md).

---

## The Max tier — Python state-machine simulator

One fully-specced family (`ecommerce_vibecoded_saas`, 22 services) with 11 composable chaos patterns. Each pattern is grounded in a real 2025-26 production incident:

| Pattern | Real-world grounding | Classification |
|---|---|---|
| `deploy_regression` | classic deploy regression class | reliability |
| `stripe_webhook_signature_regression` | Stripe webhook signature drift | reliability |
| `dependency_degradation` | Cloudflare R2 Mar 2025 / Fly.io Apr 2026 | reliability |
| `config_rollout` | Cloudflare Nov 2025 | reliability |
| `retry_storm` | Stripe Mar 2022 | reliability |
| `migration_lock` | Railway Oct 2025 | reliability |
| `rls_silent_leak` | Supabase RLS class | **security** |
| `oauth_supply_chain_pivot` | Vercel Apr 2026 (Context.ai) | **security** |
| `observability_self_denial` | Cloudflare Nov 2025 logging-storm | reliability |
| `secondary_rate_limit` | Railway Jan 2026 | reliability |
| `cdn_cache_contamination` | Railway Mar 2026 | **security** |

The simulator implements each pattern as a state-transition rule over the in-memory 22-node service graph. The same 11-action interface as Basic is reused; reward is potential-shaped over graph health. The real docker-compose stack (`sre_gym/max/families/ecommerce_vibecoded_saas/compose/`) can be brought up locally for users with Docker — see the operator notes in the family YAML.

Full design: [`docs/MAX_TIER.md`](docs/MAX_TIER.md).

---

## The HF Space UI (terminal-style Gradio)

The Space exposes a single terminal-style UI that drives all three tiers. The user picks a tier, model, and scenario; `▶ run` streams a per-tick rollout trace into the terminal. **Bring-your-own-key** for any provider — tokens live only in `gr.State` for the session, never persisted, never logged.

```
┌─────────────┬──────────────────────────┬──────────────────────────┐
│ Tier        │ Model & API config       │ Scenario picker          │
│  ◯ Basic    │ provider:  hf  ▼         │ tier:    Basic           │
│  ◯ Advanced │ HF token:  ********      │ template: memory_leak…   │
│  ◉ Max      │ model:     Qwen2.5-7B ▼  │ seed:    42              │
└─────────────┴──────────────────────────┴──────────────────────────┘
┌────────────────────────────────────────────────────────────────────┐
│ rollout terminal                                                   │
│ [00:00] === sre-gym Basic :: memory_leak_oom (seed=42) ===         │
│ [00:00] tick= 1/13 action=query_logs       reward=-0.010 …         │
│ [00:01] tick= 2/13 action=query_metrics    reward=-0.010 …         │
│ ...                                                                │
└────────────────────────────────────────────────────────────────────┘
[ ▶ run ] [ ■ stop ] [ ↻ reset ]   reward: 0.756  steps: 10  resolved: True
```

Provider routing: HF Inference Router (default), Anthropic SDK, OpenAI-compatible (Together / Groq / Fireworks / DeepSeek / OpenAI). Models per tier are curated defaults; any HF model ID or BYOK provider config also works.

---

## Training pipeline (Basic only)

Full Colab notebook: [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb). Unsloth + TRL GRPO; Qwen2.5-3B in 4-bit; LoRA r=64; SFT cold start (500 steps) → GRPO (800 steps); ~12h on a single A100, ~$30 of HF credits.

| Stage | Time | Cost |
|---|---|---|
| Seed dataset (Claude / Llama teachers) | ~2h | ~$15 API |
| SFT cold start (500 steps) | ~3h | A100 only |
| GRPO online (800 steps, K=4 rollouts) | ~6h | A100 only |
| Held-out eval (36 episodes) | ~30min | A100 only |

Comparison-table notebook: [`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) runs random / heuristic / scripted / Llama-3.3-70B / Claude Haiku / Claude Sonnet / **trained Qwen2.5-3B** against the held-out 12 procgen variants and produces the comparison plots in `eval/results/`.

---

## Frontier baselines (Basic, measured against the live Space)

| Policy | Episodes | Resolved | Mean score |
|---|---|---|---|
| Heuristic (deterministic) | 18 | 0/18 | **0.19** |
| Random (uniform) | 12 | 0/12 | **0.35** |
| Llama-3.3-70B-Versatile (Groq) | 11 | 5/11 | **0.42** |
| Llama-3.3-70B-Instruct (Fireworks) | 4 | 3/4 | **0.73** |
| Scripted-optimal baseline | 12 | 12/12 | **≤ 0.80** (CI-enforced) |
| Claude Opus 4.7 (hand-driven) | 6 | 6/6 | **0.77** |
| **Trained Qwen2.5-3B** (target) | — | — | **target ≥ 0.80** |

A 0.58-wide spread between heuristic and Claude Opus means the env discriminates capability without saturating.

---

## Tier-aware Python API

```python
from sre_gym import SREGym, Tier

# Basic: per-step or end-to-end
env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
result = env.run("memory_leak_oom__p02", seed=42)

# Advanced: episodic only
env = SREGym(tier=Tier.ADVANCED)
result = env.run("cascading_release_train", seed=1)
print(result.summary())

# Max: per-step (graph mutations) or end-to-end
env = SREGym(tier=Tier.MAX)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  app.py (Gradio terminal UI)                        │
└──────────┬─────────────────────┬────────────────────┘
           │                     │ provider routing
           ▼                     ▼
┌─────────────────────┐  ┌────────────────────────────┐
│ sre_gym (tier API)  │  │ sre_gym/ui/providers.py    │
│  ├ Tier.BASIC       │  │  HF / Anthropic / OpenAI-  │
│  ├ Tier.ADVANCED    │  │  compatible (BYOK)         │
│  └ Tier.MAX         │  └────────────────────────────┘
└──────────┬──────────┘
           │ delegates / runners
           ▼
┌────────────────────────────────────────────────────┐
│ unified_incident_env (Basic, /step + /mcp routes)  │
│ sre_gym.advanced.runner (chained Basic episodes)   │
│ sre_gym.max.runner (Python state-machine sim)      │
└────────────────────────────────────────────────────┘
```

The full architectural narrative — including the dimensional-escalation defence, OpenEnv framework integration, MCP dual-route design, judging-criteria mapping, and 45-incident postmortem corpus — lives in [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md).

Other docs:
- [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md) / [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md) / [`docs/MAX_TIER.md`](docs/MAX_TIER.md) — per-tier deep dives
- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — composable rubric, shaping, hardened ceiling
- [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) — add a 13th template
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — postmortems + benchmarking literature
- [`execution.md`](execution.md) — full operator runbook (clone → train → deploy)

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, India 2026 — by the dakshdoesdev / Madhav189 team.
