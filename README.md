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

> **Hackathon submission — OpenEnv-class, India 2026**
>
> - 📖 **Blog:** [BLOG.md](BLOG.md)
> - 🚀 **Live HF Space:** https://huggingface.co/spaces/Madhav189/sre-env
> - 💻 **GitHub:** https://github.com/Madhav-GPT/sre-env
> - 🧪 **Training notebook:** [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb)
> - 📊 **Eval results:** [`eval/results/`](eval/results/)
> - 📜 **License:** Apache 2.0

**Each tier escalates a different dimension. Triage escalates compute, Strategy escalates horizon, Operations escalates realism.** That single sentence is the load-bearing claim of the project.

The repo's centre of gravity, in priority order:

1. **The environment** — 12 incident templates × 6 procgen variants = 72 deterministic scenarios, exposed via the OpenEnv contract (`/reset` / `/step`) on a FastAPI server. Same code path serves the Gradio UI mounted at `/`.
2. **The reward rubric** — a 5-component composite that sums to exactly 1.0, with a heuristic ceiling pinned to `[0.65, 0.80]` and a scripted-expert floor at `≥0.90`, both enforced by CI invariants on every commit. Includes a calibration term inside `submit_hypothesis` that grades confident-wrong twice as harshly as hedged-wrong — a small **world-modelling** primitive: the agent has to maintain a belief over root causes and emit a calibrated confidence estimate.
3. **Coliseum** — a parallel-rollout pool server that turns the env into a lease-based HTTP service so any GRPO trainer can drive K-rollouts-per-scenario without holding a Python env per worker.
4. **Training & datasets (the honest weak point)** — an end-to-end SFT → GRPO pipeline on Qwen2.5-7B-Instruct, trained against a 120-episode trajectory corpus harvested from the env. The pipeline runs cleanly; the corpus and step budget are smaller than they need to be to break the heuristic ceiling on held-out scenarios. **The env is ready to train against; we ran out of compute before the model was.**

---

## What's in the box

| Tier | Runnable kind | Scenarios | What "running" means |
|---|---|---|---|
| **Triage** | live HTTP env | 12 templates × 6 entries each (1 base + 5 procgen) = **72 scenarios** | `/reset` + `/step` against the FastAPI server in this Docker image. The Gradio UI drives episodes end-to-end via the same routes. |
| **Strategy** | Python orchestrator | 3 reference YAML scenarios | `sre_gym.strategy.runner.run_strategy` chains Triage episodes together, threading horizon state (unresolved alerts, pending deploys, tech-debt counter, horizon-decay reward). The 28-action universe in the YAML is design spec; the runner uses the Triage 11 actions. |
| **Operations** | Python state-machine simulator | 1 family with 11 chaos patterns | `sre_gym.operations.runner.run_operations` mutates an in-memory 22-node service graph. Same Triage 11 actions. The compose stack alongside the simulator describes the topology an enterprise team would lift into a real cluster — the simulator runs without that lift. |

The escalation axis is the point: each tier hardens a different bottleneck of building SRE agents in production.

---

## Quickstart

### 5-minute local demo (no API keys, no server, no GPU)

```bash
pip install -e .
ollama pull llama3.2
python -m sre_gym.local triage worker_deploy_cascade
```

The CLI drives `UnifiedIncidentEnvironment` directly and prints per-tick reward, the 5-component score breakdown, and a final summary. See [`sre_gym/local.py`](sre_gym/local.py) for the full flag set.

### Live HF Space (Triage tier, hosted)

Open https://huggingface.co/spaces/Madhav189/sre-env. Pick a scenario and a model provider, click **▶ run eval**. Each tick streams the action, env response, reward delta, and the 5-component breakdown.

### Local server + Gradio UI

```bash
make install
make dev                                              # FastAPI + Gradio on :7860
python -m sre_gym.strategy run cascading_release_train
python -m sre_gym.operations run ecommerce_vibecoded_saas --chaos rls_silent_leak
```

The FastAPI server speaks the OpenEnv contract (`/reset /step /state /tasks /baseline /grader /status /health /metadata /schema`) plus an MCP JSON-RPC route at `/mcp`.

---

## The Triage tier — the runnable contract

12 base templates of one-incident-at-a-time scenarios; each generates 5 procgen variants for 72 scenarios total. The agent has 11 bounded actions:

```
query_logs(service)            query_metrics(service, metric)
query_dependencies(service)    query_deploys(service)
rollback_deploy(service)       restart_service(service)
isolate_service(service)       run_check(check_name)
submit_hypothesis(hypothesis)  escalate
declare_resolved
```

A successful episode looks like: `gather evidence → submit_hypothesis → rollback_deploy → restart_service → both run_check pass → declare_resolved`. Wrong rollback target, premature restart, or premature resolution all return negative reward and a typed `failure_type`.

Services live in a 4-node topology (`api-gateway / cache / database / worker`) plus an 11-service noise-decoy pool that surfaces in alerts as decoys but never in queries. Each scenario specifies a root cause, the correct rollback target, the resolution check that must pass, and the decoy traps. See [`docs/TRIAGE_TIER.md`](docs/TRIAGE_TIER.md) for the per-template skill table.

The Triage server is **the** runnable contract — Strategy and Operations chain Triage episodes; Coliseum (below) wraps Triage in a lease-based HTTP shape; the local CLI imports the Triage env in-process. Everything else is a runner shape on top.

---

## The reward rubric — the engineering crown jewel

Triage uses a **5-component rubric** that sums to exactly 1.0 — see [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md):

```
final_reward = 0.45·outcome
             + 0.20·action_validity
             + 0.15·anticheat
             + 0.10·format
             + 0.10·efficiency
             ────
                    composite ∈ [0.01, 0.99]   (clamped, public score)

  outcome          root-cause action correct + recovery confirmed
  action_validity  fraction of step() actions that are well-formed Pydantic
  anticheat        declare_resolved blocked unless ≥1 query already ran
  format           submit_hypothesis was called before declare_resolved
  efficiency       exp(-current_tick / optimal_ticks_for_template)
```

Plus per-tick *shaped* reward (the change in incident-health potential) for dense GRPO signal. Strategy and Operations reuse the Triage rubric and apply a horizon-decay factor over per-phase composites.

Two reference scores anchor the rubric and are CI-pinned:

- **Heuristic ceiling `[0.65, 0.80]`** — a naive policy that gathers evidence and submits the correct hypothesis but never remediates lands here. Enforced by `test_heuristic_ceiling_is_in_band` across all 12 templates. The 0.20 gap from 0.80 → 1.00 is the GRPO training target.
- **Scripted-expert floor `≥0.90`** — the optimal canonical solve scores ~0.94 on every template. Enforced by `test_round2_baseline_resolves`.

Adversarial cheats are first-class:

| Cheat strategy | Blocked by |
|---|---|
| `declare_resolved` before any query | `anticheat` (0.15) |
| Skip `submit_hypothesis` to save a tick | `format` (0.10) |
| Spam hypotheses to fish for partial credit | hypothesis idempotence + `action_validity` |
| Send malformed actions | `action_validity` (0.20) |
| Resolve before checks pass | `outcome` (0.45) + `premature_resolution` step penalty |

The cleverest piece is the **calibration term inside `submit_hypothesis`**:

```
hypothesis_reward = 0.04·cause_match
                  + 0.03·service_match
                  + 0.03·action_quality
                  + 0.02·calibration

calibration awards
   +1.0   confident-correct
   +0.5   hedged-correct
   -0.2   hedged-wrong
   -1.0   confident-wrong
```

The `confidence ∈ [0,1]` field is part of the structured `HypothesisPayload` Pydantic model the agent emits; the calibration sub-term reads it directly. **A model that bluffs high confidence on a wrong root cause is worse than one that hedges.** That's the world-modelling primitive — the env is grading the agent's belief, not just its prediction.

---

## Coliseum — parallel-rollout pool server

[`coliseum/`](coliseum/) wraps the Triage env in a lease-based HTTP contract so a GRPO trainer's parallel-rollout side can drive the env without holding an in-process `UnifiedIncidentEnvironment` per worker:

```
allocate(task_key)                    -> {ok: true, lease_id}
reset(lease_id, task_meta, run_ctx)   -> {ok: true, observation}
exec_tool(lease_id, tool_call)        -> {ok: true, observation}
evaluate(lease_id)                    -> {ok: true, score}
close(lease_id)                       -> {ok: true}
```

8-way concurrent rollouts on a single process via per-lease `asyncio.Lock`; a background reaper evicts idle leases after `COLISEUM_LEASE_TTL_S` so a crashed worker doesn't leak env instances.

```bash
# Boot the pool server
uvicorn coliseum.server:app --host 0.0.0.0 --port 8100

# Drive it from a trainer
export COLISEUM_BASE_URL=http://127.0.0.1:8100
```

Standard lease-pool pattern — see [`coliseum/README.md`](coliseum/README.md) for the full env-var table.

---

## Training & datasets — the honest weak point

The environment is the project. Training scripts orbit around it. We ran a real end-to-end Triage run on the day of the deadline — pipeline works, results are below the heuristic, the gap is real, and we're saying so.

### What we ran

Pipeline lives in [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb) (target: A100 80GB, ~2-3h end-to-end):

1. **SFT cold-start** — Unsloth + Qwen2.5-7B-Instruct (4-bit) + LoRA r=32, 50 steps × batch 16 on a 999-example diverse trajectory corpus. Eval perplexity 1.755 (healthy `[1.5, 3.0]` band). Saved to `outputs/qwen25_7b_sft_final/`.
2. **GRPO online** — TRL's `GRPOTrainer`, 40 steps × K=2 rollouts. Reward = composite + scenario-aware first-action bonus. Saved to `outputs/qwen25_7b_grpo_final/`.
3. **Held-out eval** — 12 `__p05` scenarios × 3 seeds = 36 episodes per policy, 5 policies compared.

### What it produced

| policy | mean | median | p25 | p75 | resolved_rate |
|---|---|---|---|---|---|
| random | 0.342 | 0.378 | 0.340 | 0.380 | 0/36 |
| **qwen25-7b-sft-only** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| **qwen25-7b-grpo** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| heuristic (queries + correct hypothesis, no remediation) | 0.704 | 0.705 | 0.703 | 0.705 | 0/36 |
| scripted-optimal | 0.938 | 0.939 | 0.937 | 0.940 | 36/36 |

Honest reading:
- SFT lifted the model 11% above random (0.342 → 0.379). Format-learning worked: the trained model produces 100% schema-valid `action_type` JSON.
- GRPO did not move the mean further on K=2 / 40 steps / 7B. The advantage signal exists but the budget is too small to overcome the heuristic plateau at 0.704.
- The env's rubric refuses to give partial credit for "looks right" output. A model that emits a polished hypothesis but never calls `rollback_deploy` is rewarded for zero remediation, and the 0.45 `outcome` slice stays empty. **The numbers don't lie about the work the model didn't do** — and that's exactly the rubric's job.

### What's bottlenecking us

The training-time weak spot is **data scale and step budget**, not the env. The 120-episode corpus + 50-step SFT + 40-step GRPO is what fits inside a hackathon weekend on one A100. The env, the rubric, and Coliseum are all sized for the real run; the trajectory corpus and the GRPO budget are not. Scaling either side (more teacher trajectories, more GRPO steps with K=4 rollouts, longer eval window) is the obvious next move and the bottleneck stops being the env.

The training scripts in [`train/`](train/) are working as written for the dataset we have — `build_corpus.py` produces a clean 60/20/20 quality-stratified split, `eval_sweep.py` drives the held-out comparison, `run_expert_collection.py` harvests teacher trajectories. They don't need rewriting; they need more data flowing through them.

---

## Two-paths agent design

The repo ships **two independent paths** to a working SRE agent. They share the env contract but trade compute for capability differently.

### Path A — verified-runbook skill (zero training)

[`skill/`](skill/) packages the env as a Claude Code skill. The agent reads scenario evidence, writes a verified runbook on a clean solve, and reads the runbook on the next attempt. No training, no GPU.

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
bash demo/run_demo.sh                                # end-to-end demo
```

12 verified-runbook drafts ship in [`skill/verified-runbooks/`](skill/verified-runbooks/) — one per Triage template. The skill validates them by re-running the env after each solve.

### Path B — GRPO-trained adapter (one A100, ~2–3h on 7B)

The training pipeline above. Path A is what an agent ships *today*; Path B is what raises the floor on the templates it sees over and over.

---

## The HF Space UI

The Gradio app at https://huggingface.co/spaces/Madhav189/sre-env is mounted at `/` of the same uvicorn process that serves `/reset` + `/step`. Three tiers selectable as cards (Triage live HTTP, Strategy chained-episode runner, Operations graph simulator). Every run streams per-tick action, env response, reward delta, and the 5-component breakdown — `out=… valid=… fmt=… anti=… eff=…`.

Provider auth is whatever the user pastes (HF token plus optional Anthropic / OpenAI / Together / Fireworks / Groq / DeepSeek key). Tokens live only on the request instance — never logged, never persisted, never echoed in error messages. CSS theme is GitHub-dark phosphor on a JetBrains Mono base; see [`app.py`](app.py) for the full styling block.

---

## Tier-aware Python API

```python
from sre_gym import SREGym, Tier

# Triage — per-step (live FastAPI) or end-to-end
env = SREGym(tier=Tier.TRIAGE)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
result = env.run("memory_leak_oom__p02", seed=42)

# Strategy — episodic only (chained Triage episodes)
env = SREGym(tier=Tier.STRATEGY)
result = env.run("cascading_release_train", seed=1)

# Operations — per-step (graph mutations) or end-to-end (state-machine simulator)
env = SREGym(tier=Tier.OPERATIONS)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

Old tier names (`Tier.BASIC`, `Tier.ADVANCED`, `Tier.MAX`) are preserved as Enum aliases so existing callers keep working; importing them emits a `DeprecationWarning`.

---

## Tests + lint

```bash
make test            # green at HEAD
ruff check .         # configured; pre-existing F401 cleanups tracked separately
openenv validate .   # green
```

The two CI invariants that keep the rubric calibrated:

- `test_heuristic_ceiling_is_in_band` — naive heuristic in `[0.65, 0.80]` on every template.
- `test_round2_baseline_resolves` — scripted-optimal `≥ 0.90` on the 6 round-2 templates.

Tighten either side and the gradient signal collapses; loosen either side and a memorising model can game the rubric without learning causality. The band is the load-bearing engineering claim.

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
│       ├─ /mcp                        JSON-RPC 2.0 dual-route │
│       ├─ /docs /redoc /openapi.json  Swagger / ReDoc         │
│       └─ /info /simple               legacy markdown landing │
│                                                              │
│  sre_gym/                                                    │
│   ├─ tier.py                Tier enum + TierConfig           │
│   ├─ env.py                 SREGym factory (delegates per t.)│
│   ├─ basic_runner.py        wrap UnifiedIncidentEnvironment  │
│   ├─ strategy/runner.py     chain Triage episodes + horizon  │
│   ├─ operations/runner.py   Python state-machine over 22 nd. │
│   ├─ ui/                    providers, router, policies      │
│   ├─ local.py               in-process CLI for Ollama models │
│   └─ exceptions.py          typed errors                     │
│                                                              │
│  coliseum/                  parallel-rollout pool server     │
│   ├─ server.py              FastAPI lease pool               │
│   └─ client.py              ArenaClient + create_arena_client│
│                                                              │
│  notebooks/                                                  │
│   └─ 01_triage_train_grpo_qwen25_7b.ipynb   SFT → GRPO pipe. │
│                                                              │
│  skill/                     Claude Code skill (Path A)       │
│   ├─ SKILL.md               agent instructions               │
│   ├─ tools/                 sre-gym HTTP client              │
│   └─ verified-runbooks/     12 per-template runbooks         │
└──────────────────────────────────────────────────────────────┘
```

Per-tier deep dives in [`docs/TRIAGE_TIER.md`](docs/TRIAGE_TIER.md) / [`docs/STRATEGY_TIER.md`](docs/STRATEGY_TIER.md) / [`docs/OPERATIONS_TIER.md`](docs/OPERATIONS_TIER.md). Reward design: [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md). Operator guide: [`execution.md`](execution.md). Architectural narrative: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Blog: [`BLOG.md`](BLOG.md).

---

## Materials

- [`openenv.yaml`](openenv.yaml) — declares the three tiers, runnable kinds, scenario counts.
- [`pyproject.toml`](pyproject.toml) — Python package, deps, entry points.
- [`docs/`](docs/) — architecture, per-tier deep dives, reward design, scenario authoring guide, references.
- [`docs/blog/`](docs/blog/) — the 6 blog assets (hero, topology, rubric donut, chaos timeline, two-paths, baselines bar).
- [`skill/`](skill/) — Claude Code skill packaging (Path A).
- [`coliseum/`](coliseum/) — parallel-rollout pool server.
- [`demo/`](demo/) — `run_demo.sh` end-to-end demo, `pitch.md` narrative.
- [`eval/`](eval/) — held-out split definition, results directory.
- [`train/data/`](train/data/) — teacher trajectories (Claude Opus + Llama-3.3-70B + scripted baselines + 120-episode v2 corpus).
- [`notebooks/`](notebooks/) — Triage SFT→GRPO training (`01_triage_train_grpo_qwen25_7b.ipynb`), eval comparison (`02_triage_eval_compare_all.ipynb`), Strategy + Operations walkthroughs.

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, India 2026 — by the Madhav-GPT / dakshdoesdev team.
