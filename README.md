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

# sre-gym — Fault-injecting SRE training env for vibe-coded SaaS

> **45% of AI-generated code ships with at least one security flaw.**
> **88% of AI-generated logging doesn't sanitize inputs.**
> **40% of AI-generated database queries are SQL-injectable.**
>
> *— Veracode (100+ LLMs, 80 vulnerability scenarios), JFrog / Snyk, Accorian — measurements across 5,600 deployed vibe-coded apps, 2025–2026.*

In July 2025, a Replit Agent deleted Jason Lemkin's SaaStr production database during an explicit code freeze. In 2025 the Tea app leaked user data through unauthenticated admin routes. The Base44 SaaS platform shipped a URI-construction bug that let unauthenticated users hit privileged endpoints. These aren't bugs — they're the new baseline. This is the fastest-shipping software segment on Earth, and it has the weakest SRE muscle of any category ever shipped.

**sre-gym** is a fault-injecting environment where an agent diagnoses vibe-coded SaaS incidents, chooses a safe remediation, verifies recovery, and declares resolved. Deterministic grading, honest world model, no hidden oracles, no gameable reward paths. Every run scores the same way twice.

- **Live:** [dakshdoesdev-sre-gym.hf.space](https://dakshdoesdev-sre-gym.hf.space) ([`/health`](https://dakshdoesdev-sre-gym.hf.space/health))
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- **Tests:** 36 passing, `openenv validate` green, drop-in OpenEnv compliance.

---

## What's inside

| | |
|---|---|
| **Env** | 6 scenario templates × 5 procgen variants = **30 live scenarios**. Typed Pydantic `Action`/`Observation`/`State`. FastAPI+WebSocket session server. |
| **Agent interface** | 11 bounded actions (query / remediate / verify). Evidence-grounded hypothesis scoring — lucky guesses don't score. |
| **Grader** | 7 deterministic dimensions. Public ceiling ~0.80. Speed-bonus for sub-optimal-tick solves. Noise-query penalty. |
| **Training** | Seed SFT dataset from Claude-as-teacher (6 episodes, 39 samples). Full Colab+Unsloth+TRL sanity notebook. OpenClaw-RL pool-server shim for async GRPO. |
| **Claude skill** | `skill/SKILL.md` + `tools/sre_gym_client.py` + `verified-runbooks/`. Successful solves write markdown runbooks the next session reads back. |

---

## 30-second demo

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm && cd sre-enginnerllm
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
./demo/run_demo.sh
```

Starts the env, solves each scenario cold, writes a runbook, re-solves to prove the loop. ~10 seconds end-to-end.

---

## Curriculum — 6 templates, 30 scenarios

Each template ships with 5 seeded procgen variants (`__p01..__p05`) that jitter metrics, deploy timestamps, and noise-service decoys. Trained agents cannot memorize a specific metric fingerprint.

| Difficulty | Scenario | 2025–26 incident pattern | Correct path |
|---|---|---|---|
| easy | `worker_deploy_cascade` | Bad worker deploy → DB crash-loop → login 502s (classic deploy cascade) | rollback worker → restart db → verify → resolve |
| medium | `db_config_rollout` | DB config push shrank connection pool 80→12 (Cloudflare Nov 2025 permissions pattern) | rollback **db** → restart db → verify |
| medium | `payment_webhook_misconfig` | Gateway deploy broke Stripe webhook signature verification. Users charged, subs inactive. | rollback **gateway** → verify |
| medium | `schema_drift_missing_migration` | Gateway deploy expects `users.plan_tier`; migration never ran in prod (Prisma/Supabase drift) | rollback **gateway** → verify |
| medium | `cache_stale_state` | Cache deploy bumped session TTL 30s→3600s. Users see cross-user state. | rollback **cache** → restart cache → verify |
| hard | `gateway_auth_rollout` | Auth-middleware rollout rejects valid logins (cf. Base44 incident shape) | rollback **gateway** → verify (no restart) |

**Noise services** — `stripe-webhook`, `sentry`, `supabase-realtime`, `openai-proxy`, `clerk-auth`, `feature-flags`, `analytics`, `email-queue`, `image-cdn`, `sessions-redis`, `vercel-edge` — surface plausibly-relevant alerts that are historically benign. They never appear in `service_health` so agents can't query them through the action schema, but they do appear in alerts as decoys. Each noise query deducts from `noise_handling_score`.

Rolling back the wrong service returns negative reward with `failure_type="wrong_remediation_target"`. Restarting before the cause is removed re-inherits the bad state. `declare_resolved` is rejected until the scenario's resolution check passes against the actual world model.

---

## Action space (11 bounded actions)

| Action | Purpose |
|---|---|
| `query_logs(service)` | Read service log stream. First query per service is free; second costs a tick. |
| `query_metrics(service, metric)` | CPU / error_rate / latency time series. |
| `query_deploys(service)` | Recent deploy history with version string + relative timestamp. |
| `query_dependencies(service)` | Causal dependency chain. |
| `rollback_deploy(service)` | Revert the most recent deploy. Negative reward if wrong target. |
| `restart_service(service)` | Restart. Rejected with `failure_type="premature_restart"` if the root cause hasn't been removed first. |
| `isolate_service(service)` | Containment. Applies but does **not** resolve — checks still have to pass. |
| `run_check(check_name)` | `database_recovery` or `end_to_end`. |
| `submit_hypothesis({root_cause, affected_services, confidence, recommended_next_action})` | Earns reward proportional to root-cause accuracy, service localization, confidence calibration, and next-action quality. Not farmable — second identical hypothesis returns 0. |
| `escalate` | No-op with step-cost. |
| `declare_resolved` | Terminal. Rejected with `failure_type="premature_resolution"` if resolution check hasn't passed. |

---

## Scoring rubric (7 dimensions, deterministic)

| Dimension | Weight | What it measures |
|---|---|---|
| `recovery_score` | 0.25 | Critical-path services are healthy, weighted per scenario. |
| `containment_score` | 0.15 | Root cause removed (0.15) or offending service isolated (0.10). |
| `verification_score` | 0.20 | `database_recovery` (+0.08) and `end_to_end` (+0.12) checks passed. |
| `impact_score` | 0.05 | User-impact reduced from baseline. |
| `efficiency_score` | 0.05 | Blast-radius budget preserved (no wasteful repeats / extra mitigations). |
| `speed_bonus` | 0.00–0.10 | Finishing under `optimal_ticks`, conditional on full verification. Skipping checks to chase speed scores *lower*. |
| `noise_handling_score` | 0.00–0.05 | Penalizes querying distractor noise services. |

Scripted-optimal baseline ceiling is hardened at **≤ 0.80** across all scenarios. Headroom is left for a trained agent that earns `speed_bonus` by finishing faster while keeping verification complete.

---

## Live deployment

The env is live as a Hugging Face Space in Docker SDK mode:

```
https://dakshdoesdev-sre-gym.hf.space
├── /health                   status probe
├── /tasks                    scenario catalog (30 scenarios)
├── /baseline                 scripted-optimal trajectory per scenario
├── /status                   current runtime + grader state
├── /reset                    OpenEnv reset
├── /step                     OpenEnv step
└── /state                    OpenEnv state snapshot
```

Direct WebSocket session client:

```python
from unified_incident_env import UnifiedIncidentEnv, UnifiedIncidentAction

with UnifiedIncidentEnv(base_url="https://dakshdoesdev-sre-gym.hf.space").sync() as env:
    obs = env.reset(scenario_id="payment_webhook_misconfig")
    obs = env.step(UnifiedIncidentAction(action_type="query_deploys", service="api-gateway"))
    print(obs.tool_output)
    # "Rolled out gateway@2026.04.24-stripe-fix 18 minutes ago (Stripe webhook handler rewrite; API version bump 2023-10-16 -> 2024-06-20)."
```

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │   sre-gym (this repo)               │
                         │                                     │
┌───────────────┐  WS    │  ┌──────────────────────────────┐   │
│ Claude Code   │───────▶│  │ unified_incident_env         │   │
│ + skill       │◀───────│  │  ├ models.py (typed API)     │   │
└───────────────┘        │  │  ├ server/environment.py     │   │
       │                 │  │  ├ server/challenge.py       │   │
       ▼                 │  │  ├ server/grader.py          │   │
 verified-runbooks/      │  │  └ tests/  ✓ 36 green        │   │
   *.md (grows over      │  └──────────────────────────────┘   │
    successful solves)   │         ▲                ▲          │
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
                          cpu-basic             distributed
```

---

## Install

**Env (local):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python -m uvicorn unified_incident_env.server.app:create_compatible_app \
  --factory --host 127.0.0.1 --port 8000
```

**Verify:**

```bash
pytest unified_incident_env/tests -q          # 36 tests
python -m openenv.cli validate .              # OpenEnv manifest check
curl http://127.0.0.1:8000/health             # {"status":"healthy"}
```

**Claude Code skill:**

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

Then in Claude Code: *"Solve the `cache_stale_state` scenario in sre-gym."* The skill drives the env via `skill/tools/sre_gym_client.py`, loads any existing runbook from `skill/verified-runbooks/`, appends a fresh runbook on any clean solve (score > 0.85).

---

## Quick start — solve a scenario

```bash
# Boot env
python -m uvicorn unified_incident_env.server.app:create_compatible_app \
  --factory --port 8000 &

# Solve a scenario via the skill client (uses verified runbooks)
export SRE_GYM_BASE_URL=http://127.0.0.1:8000
python skill/tools/sre_gym_client.py solve worker_deploy_cascade

# List all scenarios
python skill/tools/sre_gym_client.py list

# Interactive stepping
python skill/tools/sre_gym_client.py interactive payment_webhook_misconfig
```

Expected solve output:
```
[reset]  scenario=worker_deploy_cascade difficulty=easy
[step 1] action={"action_type":"query_deploys","service":"worker"}  reward=-0.01  score=0.17
[step 2] action={"action_type":"query_logs","service":"worker"}     reward=-0.01  score=0.17
...
[step 10] action={"action_type":"declare_resolved"}                  reward=+0.24  score=0.74
[done]  resolved=True  score=0.74  steps=10
```

---

## Training pipeline

The env is built to be trained against. The pipeline has four components, all included:

**1. Mixed-teacher seed dataset** (shipped in `train/data/seed_combined.jsonl`)

Two-teacher warm-start corpus built deliberately for SFT → GRPO:

| Teacher | Episodes | Mean score | Role |
|---|---|---|---|
| Claude Opus 4.7 (hand-driven via pool server) | 6 | 0.769 | Expert demos — author-optimal paths, all resolved |
| Llama-3.3-70B-Instruct via Fireworks | 4+ | 0.725 | Realistic agent — noisier, some unresolved |
| Llama-3.3-70B-Versatile via Groq free tier | growing | varies | Even noisier, higher-entropy rollouts for GRPO |

The variance between teachers is deliberate — Claude teaches format + optimal paths; Llama teaches what realistic-agent failure looks like. 78+ usable samples, growing. All 6 scenario templates covered.

**2. Parallel async trajectory collection** (`train/collect_trajectories.py`)

Async worker pool. Four drivers:
- `--driver anthropic` — Claude via Anthropic API
- `--driver fireworks` — any Fireworks-served model (Llama-3.3-70B, DeepSeek-V3.1, Kimi-K2.5)
- `--driver groq` — any Groq-served model (Llama-3.3-70B-Versatile on free tier, ~14K req/day)
- `--driver heuristic` — deterministic dumb baseline (floor)

All handle 429 `Retry-After` with jittered backoff so free-tier rate limits don't cascade into silent heuristic fallback.

```bash
python train/collect_trajectories.py \
  --env-url https://dakshdoesdev-sre-gym.hf.space \
  --scenarios all \
  --models "llama-3.3-70b-versatile" \
  --episodes-per-model 100 \
  --parallelism 3 \
  --driver groq \
  --output train/data/llama33_70b_groq_100.jsonl
```

**3. Unsloth + TRL SFT notebook** (`train/sanity_run.ipynb`)

Colab-ready. Loads Qwen3.5 4B in 4-bit via Unsloth (fallback Qwen3 4B), LoRA r=32 on 7 projection modules, runs 500 SFT steps on `seed_combined.jsonl`, pushes adapter to `dakshdoesdev/sre-gym-qwen35-4b-sft`. ~20 min on A100 40GB. Inference cell validates JSON action format compliance.

**4. GRPO online notebook** (`train/grpo_run.ipynb`)

Colab-ready. Loads the SFT LoRA, boots our OpenClaw pool server on the same VM, runs 300 GRPO steps. Each step samples a scenario and rolls out K=4 trajectories with the current policy; reward = `env.evaluate()['score']` (deterministic scalar from grader, no PRM or LLM-as-judge); group-relative advantages applied via simple policy gradient. Pushes to `dakshdoesdev/sre-gym-qwen35-4b-grpo` every 25 steps. Wandb-logged training curve.

**5. Eval sweep** (`train/eval_sweep.py`)

Comparison-table generator. Runs N episodes per scenario per policy against a live env, writes JSONL + summary. Supports `random`, `heuristic`, `groq`, `fireworks`, `anthropic` policies. The trained model's numbers come from the GRPO notebook's final held-out eval cell (matches the same schema).

```bash
python train/eval_sweep.py \
  --env-url https://dakshdoesdev-sre-gym.hf.space \
  --scenarios all \
  --policies random,heuristic,groq \
  --groq-model llama-3.3-70b-versatile \
  --episodes-per-scenario 5 \
  --output train/data/eval_sweep.jsonl
```

Full Friday plan: 2000 Claude-teacher trajectories → Qwen3.5 4B SFT cold start → OpenClaw-RL GRPO run against the pool server → 100-episode eval sweep across {random, untrained-3B, Haiku, Sonnet, trained-3B} → comparison table.

---

## OpenClaw-RL integration (`openclaw_integration/`)

Drop-in lease-based pool server compatible with [Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)'s async GRPO trainer:

```bash
python -m uvicorn openclaw_integration.pool_server:app --port 8100
```

```
POST /allocate    {task_key}                → {lease_id}
POST /reset       {lease_id, scenario_id}   → {observation}
POST /exec_tool   {lease_id, tool_call}     → {observation}
POST /evaluate    {lease_id}                → {score}
POST /close       {lease_id}                → {ok}
GET  /healthz                               → {ok, active_leases, scenarios}
```

`asyncio.Lock` per lease, TTL reaper for stale sessions, automatic lease cleanup on close. Mirrors the `OpenClaw-RL/terminal-rl/remote/pool_server.py` contract. `openclaw_integration/generate_with_sre.py` is an import-patch wrapper for their `terminal-rl/generate.py` — three-file shim, no edits to OpenClaw-RL internals.

---

## Project layout

```
sre-enginnerllm/
├── unified_incident_env/         # env core
│   ├── models.py                 # typed Pydantic Action / Observation / State
│   ├── client.py                 # session-aware WebSocket client
│   ├── server/
│   │   ├── app.py                # FastAPI + OpenEnv wiring
│   │   ├── environment.py        # world-state sim, recipe-driven remediation
│   │   ├── challenge.py          # 6 scenario templates + procgen + baselines
│   │   └── grader.py             # 7-dim deterministic scoring
│   └── tests/test_environment.py # 36 tests
├── skill/                        # Claude Code skill
│   ├── SKILL.md                  # frontmatter + investigation rules
│   ├── tools/sre_gym_client.py   # CLI: list / solve / interactive / record-runbook
│   └── verified-runbooks/*.md    # append-only knowledge base
├── train/                        # training pipeline
│   ├── sanity_run.ipynb          # Colab+Unsloth+TRL 200-step SFT sanity
│   ├── collect_trajectories.py   # parallel async Claude-driven collection
│   ├── compile_claude_seed.py    # event-log → canonical JSONL
│   ├── requirements-train.txt    # pinned Unsloth / TRL / wandb / anthropic
│   └── data/                     # claude_seed.jsonl (39 samples) + per-episode provenance
├── openclaw_integration/         # OpenClaw-RL drop-in shim
│   ├── pool_server.py            # FastAPI lease server
│   ├── sre_env_client.py         # async client mirroring terminal-rl interface
│   ├── generate_with_sre.py      # import-patch wrapper
│   └── README.md                 # launch instructions against OpenClaw-RL
├── demo/
│   ├── run_demo.sh               # 10s end-to-end demo
│   └── pitch.md                  # pitch script + Q&A cheat sheet
├── deploy/
│   └── push_to_hf.sh             # one-command HF Space deploy
├── inference.py                  # OpenAI-client baseline for submission
├── openenv.yaml                  # OpenEnv manifest
├── Dockerfile                    # HF Space (cpu-basic) image
└── README.md                     # this file
```

---

## Testing

```bash
pytest unified_incident_env/tests -q       # 36 tests
python -m openenv.cli validate .           # OpenEnv spec compliance
docker build -t sre-gym:v2 .               # HF Space image build
```

CI contract: every template must pass its own baseline resolution test, wrong-rollback-target must be penalized, `declare_resolved` must be rejected without checks, baseline ceiling must stay under 0.80, every scenario must expose at least one noise alert.

---

## Why this is a real research contribution

Existing SRE benchmarks have blind spots that this env deliberately fills:

| Benchmark | Shape | Gap this fills |
|---|---|---|
| [Rootly-AI-Labs/SRE-skills-bench](https://github.com/Rootly-AI-Labs/SRE-skills-bench) | ICML 2025 workshop, MCQ eval | **Not trainable.** We're a dense-reward RL env, they're static eval. |
| [agentkube/SRE-bench](https://github.com/agentkube/SRE-bench) | SWE-bench-style, real K8s | **Requires K8s cluster.** We run on cpu-basic HF Space. |
| [IBM/ITBench-SRE-Agent](https://github.com/IBM/ITBench-SRE-Agent) | K8s + CrewAI | Framework-coupled. We're framework-agnostic via OpenEnv. |
| [bugraid-ai/opensre-tools](https://github.com/bugraid-ai/opensre-tools) | "open RL env" | Generic infra failures. We specialize in **vibe-coded SaaS** specifically — the fastest-growing software category. |
| [microsoft/sre-agent](https://github.com/microsoft/sre-agent) | Azure internal | Not open infrastructure. |

**Positioning:** *the only OpenEnv-native incident-response env with a drop-in OpenClaw-RL training shim, vibe-coded-SaaS-specific failure modes grounded in 2025–26 incidents, deterministic honest grading, and a public seed dataset of frontier-model trajectories.*

---

## Roadmap

- **v1 (shipped):** 6 scenario templates × 5 procgen variants, 11-action space, 7-dim grader, Claude Code skill, OpenClaw-RL shim, Claude-teacher seed dataset, live HF Space.
- **v1.1 (hackathon window):** Scale to 1000+ procgen scenarios; collect 2000+ Claude trajectories; SFT cold-start Qwen3.5 4B; GRPO training run; comparison table across {random, untrained-3B, Haiku, Sonnet, trained-3B}.
- **v2 (post-hackathon):** Expanded action space (18–22 actions) for RLS audits, cache-header inspection, env-var diff, bundle scans. Evidence-provenance grader dimension (LLM-as-judge checks each hypothesis against actually-queried evidence). Open release of the trained `dakshdoesdev/sre-gym-qwen35-4b` model.

---

## Related research

- **danluu/post-mortems** — canonical incident corpus ([github.com/danluu/post-mortems](https://github.com/danluu/post-mortems))
- **Veracode 2025 AI code security study** — 45% of AI-generated code has security flaws (n = 100+ LLMs, 80 scenarios)
- **JFrog / Snyk** — ~40% of AI-generated DB queries are SQL-injectable
- **Accorian** — 88% of AI-generated logging unsafe, 86% input-validation contains XSS errors
- **Replit / SaaStr incident, July 2025** — Agent deleted production DB during code freeze
- **Cloudflare Nov 2025** — bot-detection permissions regression (canonical config-rollout pattern)
- **Gen-Verse/OpenClaw-RL** — async GRPO-on-next-state RL framework (training integration target)

---

## License

Apache 2.0. See `LICENSE` (TODO: add — currently inherits `license: apache-2.0` from HF Space frontmatter and `pyproject.toml`).

Built for the OpenEnv-class hackathon, April 25–26 2026, by [@dakshdoesdev](https://github.com/dakshdoesdev).
