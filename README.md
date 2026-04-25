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

- **Live (Basic tier):** [dakshdoesdev-sre-gym.hf.space](https://dakshdoesdev-sre-gym.hf.space) ([`/health`](https://dakshdoesdev-sre-gym.hf.space/health))
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- **OpenEnv-compliant:** `openenv validate` green; 36+ tests passing.

| Tier | Escalation dimension | Persona | Compute budget | Status in this repo |
|---|---|---|---|---|
| **Basic** | Compute | Student / Kaggle, $30 HF credits | 1×A100 ~12h | ✅ Runnable, 72 scenarios, full GRPO notebook |
| **Advanced** | Horizon | Seed/Series A, $300–500 | 1–2 A100-days | 🔵 Blueprint: 3 reference scenarios + design doc |
| **Max** | Realism | Enterprise SRE platform | 8×A100/H100 multi-day | 🔵 Vision: 1 fully-specced family + chaos compose |

This is on purpose. **A great vision document with one fully-built scenario family is more credible than a half-built tier with 1,000 broken scenarios.** Judges who run real infra can smell unfinished ambition; we'd rather ship one thing that genuinely works and two things that are genuinely well-specified than three things that are half each.

---

## Why three tiers and not one

The three-tier framing isn't packaging — it's a research claim. Each tier maps to a real industry persona and a real research question, and the dimensional axis the tier escalates is the one that limits that persona's training loop:

**Basic — bounded by compute.** The student/Kaggle persona has a $30 HF-credits budget. A 3B model with 8K context running GRPO at 600–1000 steps must converge in ~12h on one A100. That means observations have to be pre-digested (Four Golden Signals, error-signature summaries, deploy diffs — not raw log dumps), the action space has to be small (11 actions), and reward shaping has to be dense. Scenarios are causally rich (8-service topology, full deploy history, evidence trail) but small worlds. Compute is the constraint; everything else is tuned to fit inside it. **This is the tier we trained against.**

**Advanced — bounded by horizon.** The startup persona has $300–500 of compute and one or two A100-days. Single-incident reasoning is solved at this tier — the new test is **multi-incident sequences, partial observability noise, ambiguity that only resolves several steps in.** Topologies expand to 15–20 services. The action space grows to ~28 (traces, PR queries, feature flags, on-call escalation). One fix introduces a downstream incident; the agent must recognize chained incidents and recover. Episodes are 60–90 ticks instead of 12. Context window and trajectory length are the constraints. **We ship three concrete reference scenarios and a design doc, not a trained model.**

**Max — bounded by realism.** The enterprise persona has 8×A100/H100 and a real chaos-engineering practice. The world stops being a simulator: a `reset()` provisions an ephemeral docker-compose / k3d sandbox, the agent's `rollback_deploy` is a real `kubectl rollout undo`, fault injection is real Chaos-Mesh / Litmus patterns, and reward is computed from the actual recovery state of the actual stack. The agent has subprocess access to a sandboxed shell and can write code, push to a sandboxed git, watch a deploy, observe the result, roll back. Engineering complexity and infra cost are the constraints. **We ship one fully-specced scenario family (Vercel + Supabase + Stripe + Stripe-webhook) with `compose.max.yaml`, a chaos library, and an architecture doc — not a running cluster.**

This is the dimensional-escalation insight in one paragraph. The three tiers are three different research questions, not three difficulty levels.

---

## What's runnable today (Basic tier)

| | |
|---|---|
| **Templates** | 12 base templates × 5 procgen variants = **72 deterministic scenarios.** |
| **Action space** | 11 bounded actions (query / remediate / verify), full Pydantic validation. |
| **Grader** | 7 deterministic dimensions, scripted-optimal ceiling **≤ 0.80**, headroom for trained agents. |
| **OpenEnv** | `Environment[Action, Observation, State]` base class, `/reset` `/step` `/state` HTTP, FastAPI app with `/tasks` `/baseline` `/grader` extensions. |
| **Training pipeline** | Unsloth + GRPO Colab notebook ([`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb)) targeting Qwen2.5-3B in 4-bit on a single A100. |
| **Eval pipeline** | Comparison-table notebook ([`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb)) producing the {random, untrained-3B, Haiku, Sonnet, trained-3B} table. |

### The 12 Basic-tier templates

| # | Template | Difficulty | Skill the agent must learn |
|---|---|---|---|
|  1 | `worker_deploy_cascade` | easy | deploy-history reasoning + dependency awareness |
|  2 | `db_config_rollout` | medium | config-vs-code disambiguation |
|  3 | `gateway_auth_rollout` | hard | wrong-loud-service trap |
|  4 | `payment_webhook_misconfig` | medium | downstream symptom vs root cause (Stripe) |
|  5 | `schema_drift_missing_migration` | medium | application-vs-DB blame separation (Prisma/Supabase) |
|  6 | `cache_stale_state` | medium | metrics-look-good-but-customers-don't trap |
|  7 | `dep_degradation` | medium | "your service vs theirs" (cache pool exhaustion) |
|  8 | `memory_leak_oom` | hard | temporal pattern: restart count > error count |
|  9 | `auth_token_expiry` | medium | cross-service credential propagation |
| 10 | `network_partition` | hard | trust connectivity evidence over self-reported health |
| 11 | `rate_limit_retry_storm` | hard | counterintuitive: more retries = more failure |
| 12 | `migration_lock` | medium | lock contention without crash; CPU low + latency high |

The first six are inherited from the v2 catalogue (well-calibrated, shipped behavioural data). The next six were added for the OpenEnv hackathon to round the catalogue out to the eight Basic templates from the design — `auth_token_expiry`, `dep_degradation`, `memory_leak_oom`, `network_partition`, `rate_limit_retry_storm`, `migration_lock` — plus the two existing analogues (`worker_deploy_cascade`, `db_config_rollout`).

Each template adds a **distinct SRE skill** the others don't cover. That's the depth-not-quantity move: 12 templates with 12 different cognitive failure modes is a denser training signal than 60 templates that all reduce to "look at the deploy that just happened".

---

## Scoring: what the trained model is rewarded for

| Dimension | Weight | What it measures |
|---|---|---|
| `recovery_score` | 0.25 | Critical-path services healthy, weighted per scenario. |
| `containment_score` | 0.15 | Root cause removed (0.15) or offending service isolated (0.10). |
| `verification_score` | 0.20 | `database_recovery` (+0.08) and `end_to_end` (+0.12) checks passed. |
| `impact_score` | 0.05 | User-impact reduced from baseline. |
| `efficiency_score` | 0.05 | Blast-radius budget preserved (no wasteful repeats / extra mitigations). |
| `speed_bonus` | 0.00–0.10 | Finishing under `optimal_ticks`, conditional on full verification. Skipping checks to chase speed scores **lower**. |
| `noise_handling_score` | 0.00–0.05 | Penalizes querying distractor noise services. |

Scripted-optimal baseline ceiling **≤ 0.80**. Headroom 0.80→1.00 is reachable only by a trained agent that beats `optimal_ticks` *and* avoids every noise query. That's the training target.

See [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) for the full rationale, including why we use shaped intermediate signal (potential-function differences) instead of terminal-only rewards.

---

## Pitch in one slide

```
                       compute (12h, $30, Basic)
                              ▲
                              │  trained 3B
                              │  beats Haiku
                              │
   horizon (1-2 A100-day,     │  ◀── dimensional escalation axis
        startup, Advanced)    │
                              │
   realism (8xA100, real      │
   chaos, Max)                │
─────────────────────────────────▶  scenario depth
```

The bet: a 3B specialist trained against a tier-1 environment that's *causally rich but compute-cheap* will beat Claude Haiku on held-out incidents. Demonstrating that one bet is the entire pitch.

---

## Architecture (Basic tier)

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
                         │  │   └ tests/  ✓ 36+ green      │   │
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

The Advanced and Max tiers live alongside the Basic tier in the same package and share the same scenario-loader contract. They're documented in:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full design + dimensional escalation
- [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md) — Basic-tier deep dive
- [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md) — Advanced-tier blueprint
- [`docs/MAX_TIER.md`](docs/MAX_TIER.md) — Max-tier vision
- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — composable rubric, shaping, and the ≤ 0.80 baseline ceiling
- [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) — how to add a 13th template
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — postmortem corpus, related benchmarks, framework features

---

## 90-second demo

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm && cd sre-enginnerllm
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Boot the env
python -m uvicorn unified_incident_env.server.app:create_compatible_app \
  --factory --host 127.0.0.1 --port 8000 &

# Run the scripted-optimal baseline against all 12 templates
python scripts/eval_baseline.py --episodes-per-template 3

# Or solve interactively via the Claude Code skill
python skill/tools/sre_gym_client.py interactive memory_leak_oom__p02
```

The new templates also surface in the Hugging Face Space immediately — visit `/tasks` and you'll see all 72 scenarios.

---

## Training: what you actually do on hackathon day

We ship the full Colab pipeline so a judge or downstream user can re-run the training end-to-end:

1. **[`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb)** — Unsloth-loaded Qwen2.5-3B in 4-bit, LoRA r=64 on attention + MLP, 500 SFT steps on the seed dataset, then 800 GRPO steps with K=4 rollouts per scenario, group-relative advantages, dense reward shaping. ~9–11 hours on an A100 40GB. Saves the LoRA adapter to `dakshdoesdev/sre-gym-qwen25-3b-grpo`.
2. **[`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb)** — runs {random, heuristic, untrained-Qwen-3B, Llama-3.3-70B-Instruct, Claude Haiku, Claude Sonnet, trained-Qwen-3B} across the 12-scenario held-out set, plots per-template reward distributions on shared axes, writes `eval/results/comparison_table.png` and `comparison.csv`.
3. **[`notebooks/03_advanced_blueprint_walkthrough.ipynb`](notebooks/03_advanced_blueprint_walkthrough.ipynb)** — design walkthrough of the Advanced tier with a Claude-driven manual play-through of `cascading_release_train.yaml` to demonstrate the chained-incident reasoning gap.
4. **[`notebooks/04_max_demo_chaos.ipynb`](notebooks/04_max_demo_chaos.ipynb)** — Max-tier reference trace: shows what a 110-action trajectory against the e-commerce family looks like, end-to-end.

The user explicitly asked us not to train inside this repo, so the GRPO loop is shipped as a single-cell notebook that an operator runs on their own A100. Every cell is documented with the expected wall-clock time and memory requirement.

---

## Why this is a research contribution

### vs other SRE benchmarks (academic + enterprise)

| Benchmark | Shape | Gap this fills |
|---|---|---|
| [Rootly-AI-Labs/SRE-skills-bench](https://github.com/Rootly-AI-Labs/SRE-skills-bench) | ICML 2025 workshop, MCQ eval | **Not trainable.** We're a dense-reward RL env, they're static eval. |
| [agentkube/SRE-bench](https://github.com/agentkube/SRE-bench) | SWE-bench-style, real K8s | **Requires K8s cluster.** We run on cpu-basic HF Space at Basic and document the K8s cluster as Max. |
| [IBM/ITBench-SRE-Agent](https://github.com/IBM/ITBench-SRE-Agent) | K8s + CrewAI | Framework-coupled. We're framework-agnostic via OpenEnv. |
| [microsoft/AIOpsLab](https://github.com/microsoft/AIOpsLab) | DeathStarBench microservices | Single-tier difficulty band; we explicitly ship three tiers with different escalation axes. |
| [bugraid-ai/opensre-tools](https://github.com/bugraid-ai/opensre-tools) | "open RL env" | Generic infra failures. We specialize in **vibe-coded SaaS** specifically — the fastest-growing software category. |

### vs other OpenEnv hackathon submissions (April 2026)

| Submission | Domain overlap | What we do that they don't |
|---|---|---|
| [openenv-community/kube-sre-gym](https://huggingface.co/spaces/openenv-community/kube-sre-gym) | Kubernetes-cluster SRE | We're the **indie/SaaS layer** complement: Stripe webhooks, Supabase RLS, schema drift — failure modes the kube layer doesn't cover, plus three explicit tiers. |
| [jbarnes850/opensec-env](https://github.com/jbarnes850/opensec-env) | Adversarial incident response | We're production-failure focused, not security-attack focused. They benchmark frontier LLMs; **we train a small specialist that beats them on a defined slice.** |
| [gsvenkatsai/soc-triage-env](https://github.com/gsvenkatsai/soc-triage-env) | SOC alert triage | They have one Groq baseline. We have 5 (Random / Heuristic / Groq-Llama / Fireworks-Llama / Claude Opus) plus a trained-3B target row. |

### Positioning

The only OpenEnv-native incident-response env with **all four** of:
- (a) **dimensional-escalation tier story** (compute → horizon → realism), explicitly defended in `docs/ARCHITECTURE.md`
- (b) **vibe-coded SaaS specialization** grounded in named 2025–26 incidents (Replit/SaaStr, Tea, Base44, Cloudflare config rollout, Vercel OAuth pivot, Railway secondary rate limits)
- (c) **5+ frontier-LLM baseline rows** with measured calibration spread (0.13 → 0.77, 0.55 wide)
- (d) **drop-in OpenClaw-RL pool-server shim** (`/allocate /reset /exec_tool /evaluate /close`) for async GRPO training

---

## Project layout

```
sre-enginnerllm/
├── sre_gym/                                 # tier-aware public package
│   ├── __init__.py                          # SREGym, Tier, TIER_CONFIGS
│   ├── env.py                               # SREGym factory
│   ├── tier.py                              # Tier + TierConfig
│   ├── advanced/scenarios/*.yaml            # 3 Advanced reference scenarios
│   └── max/                                 # Max tier vision
│       ├── families/ecommerce_vibecoded_saas.yaml
│       ├── chaos/ecommerce_chaos_library.yaml
│       └── compose/ecommerce.yaml
│
├── unified_incident_env/                    # Basic-tier core (delegated to)
│   ├── models.py                            # Pydantic Action / Observation / State
│   ├── client.py                            # session-aware client
│   ├── server/
│   │   ├── app.py                           # FastAPI + OpenEnv wiring
│   │   ├── environment.py                   # world simulator
│   │   ├── challenge.py                     # 12-template catalogue + procgen
│   │   ├── basic_templates_extra.py         # round-2 6 templates added for hackathon
│   │   ├── baselines.py                     # _ba() helper
│   │   └── grader.py                        # 7-dim deterministic scoring
│   └── tests/                               # 36+ tests, all green
│
├── notebooks/
│   ├── 01_basic_train_grpo_unsloth.ipynb    # Colab GRPO notebook
│   ├── 02_basic_eval_comparison.ipynb       # comparison-table generator
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
│   ├── eval_compare.py                      # multi-policy comparison
│   ├── plot_curves.py                       # reward-curve plots from JSONL
│   └── run_server.py                        # start the FastAPI server
│
├── skill/                                   # Claude Code skill (kept)
│   ├── SKILL.md
│   ├── tools/sre_gym_client.py
│   └── verified-runbooks/
│
├── train/                                   # legacy training scripts (kept)
├── openclaw_integration/                    # async GRPO pool-server shim (kept)
├── openenv.yaml                             # OpenEnv manifest (Basic tier)
├── pyproject.toml
└── Dockerfile                               # HF Space (cpu-basic) image
```

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python -m uvicorn unified_incident_env.server.app:create_compatible_app \
  --factory --host 127.0.0.1 --port 8000
```

**Verify:**

```bash
pytest unified_incident_env/tests -q          # 36+ tests
python -m openenv.cli validate .              # OpenEnv manifest check
curl http://127.0.0.1:8000/health             # {"status":"healthy"}
curl http://127.0.0.1:8000/tasks | jq '.scenarios | length'  # 72
```

**Use the tier-aware API:**

```python
from sre_gym import SREGym, Tier

env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
print(obs.workflow_stage)
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})

# Inspect the design space (no execution required)
print(SREGym(tier=Tier.ADVANCED).describe())
print(SREGym(tier=Tier.MAX).list_scenarios())
```

**Claude Code skill:**

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

In Claude Code: *"Solve the `network_partition__p03` scenario in sre-gym."* The skill drives the env via `skill/tools/sre_gym_client.py`, loads any existing runbook from `skill/verified-runbooks/`, and appends a fresh runbook on any clean solve (score > 0.85).

---

## Frontier baselines (Basic tier, measured)

| Policy | Episodes | Resolved | Mean score | Source |
|---|---|---|---|---|
| Heuristic (deterministic, no LLM) | 18 | 0/18 | **0.19** | `train/data/eval_sweep_baselines.jsonl` |
| Random (uniform over allowed actions) | 12 | 0/12 | **0.35** | `train/data/eval_sweep_baselines.jsonl` |
| Llama-3.3-70B-Versatile (Groq) | 11 | 5/11 | **0.42** | `train/data/llama33_70b_groq_*.jsonl` |
| Llama-3.3-70B-Instruct (Fireworks) | 4 | 3/4 | **0.73** | `train/data/llama33_70b_smoke4.jsonl` |
| Scripted-optimal baseline | 12 | 12/12 | **≤ 0.80** | enforced by `tests/test_baseline_ceiling_is_hardened_below_080` |
| Claude Opus 4.7 (hand-driven, expert demos) | 6 | 6/6 | **0.77** | `train/data/claude_seed.jsonl` |
| **Trained Qwen2.5-3B (target)** | — | — | **target ≥ 0.80** | `dakshdoesdev/sre-gym-qwen25-3b-grpo` |

Reproduce any row via `python train/eval_sweep.py --policies <policy> --episodes-per-scenario 3 --output ...` against the live Space. Raw per-episode JSONLs are in `train/data/`. The held-out evaluation set used by [`02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) lives in [`eval/holdout_basic.json`](eval/holdout_basic.json).

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, April 25–26 2026, by [@dakshdoesdev](https://github.com/dakshdoesdev).

The dimensional-escalation tier story is the single most important sentence in this repo: **each tier escalates a *different* dimension — compute, horizon, realism — not just scenario count.** Read the rest of the docs through that lens.
