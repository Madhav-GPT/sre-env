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

**Each tier escalates a different dimension. Triage escalates compute, Strategy escalates horizon, Operations escalates realism.** That single sentence is the load-bearing claim of the project.

- **Live HF Space:** [huggingface.co/spaces/Madhav189/sre-env](https://huggingface.co/spaces/Madhav189/sre-env)
- **Repo:** [github.com/dakshdoesdev/sre-enginnerllm](https://github.com/dakshdoesdev/sre-enginnerllm)
- **OpenEnv manifest:** [`openenv.yaml`](openenv.yaml) — single source of truth for which tier is runnable in which mode.
- **Tests:** 203 collected via `pytest --collect-only -q`.

---

## What's in the box

| Tier | Runnable kind | Scenarios | What "running" means |
|---|---|---|---|
| **Triage** | live HTTP env | 12 templates × 6 entries each (1 base + 5 procgen) = **72 scenarios** | `/reset` + `/step` against the FastAPI server in this Docker image. The Gradio UI drives episodes end-to-end via the same routes. |
| **Strategy** | Python orchestrator | 3 reference YAML scenarios | `sre_gym.strategy.runner.run_strategy` chains Triage episodes together, threading horizon state (unresolved alerts, pending deploys, tech-debt counter, horizon-decay reward). The 28-action universe declared in the YAML is design spec; the runner uses the Triage 11 actions. |
| **Operations** | Python state-machine simulator | 1 family with 12 chaos patterns | `sre_gym.operations.runner.run_operations` mutates an in-memory 22-node service graph. Same Triage 11 actions. The compose stack under `sre_gym/operations/compose/` is a topology spec — the simulator uses graph mutations rather than container orchestration. |

The escalation axis is the point: each tier hardens a different bottleneck of building SRE agents in production.

---

## Quickstart

### 5-minute local demo (no API keys, no server)

```bash
# 1. Install (one-time)
pip install -e .
ollama pull llama3.2

# 2. Run a Triage scenario in-process against a local model
python -m sre_gym.local triage worker_deploy_cascade
```

That's it — no FastAPI server, no HF token, no GPU. The CLI drives `UnifiedIncidentEnvironment` directly and prints per-tick reward, per-component score breakdown, and a final summary. See [`sre_gym/local.py`](sre_gym/local.py) for the full flag set.

### Live HF Space (Triage tier, hosted)

Open [the Space](https://huggingface.co/spaces/Madhav189/sre-env). Pick a scenario and a model provider, click run. Each tick shows the action, env response, reward delta, and the 5-component breakdown.

### Local server + Gradio UI

```bash
make install
make dev                                              # FastAPI + Gradio on :8000
python -m sre_gym.strategy run cascading_release_train
python -m sre_gym.operations run ecommerce_vibecoded_saas --chaos rls_silent_leak
```

The FastAPI server speaks the OpenEnv contract (`/reset /step /state /tasks /baseline /grader /status /health /metadata /schema`) plus an MCP JSON-RPC route at `/mcp`.

---

## The Triage tier

12 base templates of one-incident-at-a-time scenarios; each generates 5 procgen variants for 72 scenarios total. The agent has 11 bounded actions:

```
query_logs(service)            query_metrics(service, metric)
query_dependencies(service)    query_deploys(service)
rollback_deploy(service)       restart_service(service)
isolate_service(service)       run_check(check_name)
submit_hypothesis(hypothesis)  escalate
declare_resolved
```

Services live in a 4-node topology (`api-gateway / cache / database / worker`) with a noise-service pool that surfaces in alerts as decoys. Each scenario specifies a root cause, the correct rollback target, the resolution check that must pass, and the decoy traps. See [`docs/TRIAGE_TIER.md`](docs/TRIAGE_TIER.md) for the per-template skill table.

The Triage server is **the** runnable contract — Strategy and Operations chain Triage episodes; the Coliseum pool server (below) wraps Triage in a lease-based HTTP shape; the local CLI imports the Triage env in-process. Everything else is a runner shape on top.

---

## The Strategy tier

A Python orchestrator that chains Triage episodes into multi-phase incidents with persistent horizon state. Each phase invokes the Triage env on a different template; unresolved alerts and pending deploys carry forward, and the per-phase composite reward decays into a final horizon-weighted score.

```python
from sre_gym import SREGym, Tier
env = SREGym(tier=Tier.STRATEGY)
result = env.run("cascading_release_train", seed=1)
print(result.final_reward, result.success, len(result.phases))
```

3 reference scenarios live in [`sre_gym/strategy/scenarios/`](sre_gym/strategy/scenarios/). The YAML schema declares a richer 28-action universe as design spec; the runner uses the Triage 11 actions. See [`docs/STRATEGY_TIER.md`](docs/STRATEGY_TIER.md).

---

## The Operations tier

A state-machine simulator over a 22-node `ecommerce_vibecoded_saas` service graph. Chaos patterns inject faults via state-transition rules; the agent uses the same Triage 11 actions to detect and remediate. The `compose/` YAML alongside the simulator describes the topology that an enterprise platform team would lift into a real cluster — the simulator runs without that lift.

```python
env = SREGym(tier=Tier.OPERATIONS)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

12 chaos patterns covering data integrity, RLS leaks, certificate expiry, schema drift, and credential rotation. See [`docs/OPERATIONS_TIER.md`](docs/OPERATIONS_TIER.md).

---

## The reward model

Triage uses a **5-component rubric** that sums to exactly 1.0 — see [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md):

```
outcome          0.45    root-cause action correct + recovery confirmed
action_validity  0.20    fraction of step() actions that are well-formed
format           0.10    submit_hypothesis was called before declare_resolved
anticheat        0.15    declare_resolved blocked unless ≥1 query-action ran
efficiency       0.10    exp(-current_tick / optimal_ticks_for_template)
                 ----
composite        1.00    public score, clamped to [0.01, 0.99]
```

Plus per-tick *shaped* reward (the change in incident-health potential) for dense GRPO signal. Strategy and Operations reuse the Triage rubric and apply a horizon-decay factor over per-phase composites.

Two reference scores anchor the rubric:

- **Heuristic ceiling (`[0.65, 0.80]`)** — a naive policy that gathers evidence and submits the correct hypothesis but never remediates lands here. Enforced by `test_heuristic_ceiling_is_in_band` across all 12 templates. The 0.20 gap from 0.80 → 1.00 is the GRPO training target.
- **Scripted-expert reference (`≥ 0.90`)** — the optimal canonical solve (`queries → hypothesis → rollback → run_check → declare_resolved`) scores ~0.94 on every template. Enforced by `test_round2_baseline_resolves`.

Adversarial cheats are first-class:

| Cheat strategy | Blocked by |
|---|---|
| `declare_resolved` before any query | `anticheat` (0.15) |
| Skip `submit_hypothesis` to save a tick | `format` (0.10) |
| Spam hypotheses to fish for partial credit | hypothesis idempotence + `action_validity` |
| Send malformed actions | `action_validity` (0.20) |
| Resolve before checks pass | `outcome` (0.45) + `premature_resolution` step penalty |

---

## Two-paths agent design

The repo ships **two independent paths** to a working SRE agent. They share the env contract but trade compute for capability differently.

### Path A — verified-runbook skill (zero training)

[`skill/`](skill/) packages the env as a Claude Code skill. The agent reads scenario evidence, writes a verified runbook on a clean solve, and reads the runbook on the next attempt. No training, no GPU — just structured retrieval over solved trajectories.

```bash
# Install the skill globally (one-time)
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"

# Or run the end-to-end demo
bash demo/run_demo.sh
```

12 verified-runbook drafts ship in [`skill/verified-runbooks/`](skill/verified-runbooks/) — one per Triage template. The skill validates them by re-running the env after each solve.

### Path B — GRPO-trained adapter (one A100, ~12h)

The training pipeline lives in [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb): seed-data build (~200 trajectories from a teacher), SFT cold-start (Unsloth + Qwen2.5-3B + LoRA), and 800 steps of GRPO with K=4 rollouts per scenario. The Coliseum pool server ([`coliseum/`](coliseum/README.md)) is the parallel-rollout side — point any GRPO trainer at `COLISEUM_BASE_URL` and the env runs the lease-pool contract.

Eval comparison lives in [`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb); held-out split is the `__p05` procgen variant of every template (12 scenarios) — pinned by deterministic seeding.

The two paths are complementary, not competing. The runbook skill is what an agent ships *today*; the trained adapter is what raises the floor below the heuristic.

---

## Frontier-model baselines (Triage, 5-component rubric)

Recomputed under the rubric live at HEAD:

| Policy | Episodes | Resolved | Mean score |
|---|---|---|---|
| Random (uniform over allowed actions) | 36 | 0/36 | **0.417** |
| Naive heuristic (queries + correct hypothesis, no remediation) | 12 | 0/12 | **0.749** |
| Scripted-optimal baseline | 12 | 12/12 | **0.938** |

The random / heuristic / scripted numbers are reproduced by the snippet in [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) §5. Frontier-model baselines (Llama-3.3-70B, Claude Opus 4.7, etc.) ship as raw trajectories in [`train/data/`](train/data/) and are scored by replaying through the new grader; see [`eval/results/README.md`](eval/results/README.md) for the planned comparison sweep.

---

## Coliseum — parallel-rollout pool server

[`coliseum/`](coliseum/README.md) wraps the Triage env in a lease-based HTTP contract (`allocate / heartbeat / reset / exec_tool / evaluate / close`) so a GRPO trainer's parallel-rollout side can drive the env without holding an in-process `UnifiedIncidentEnvironment` per worker. 8-way concurrent rollouts on a single process via per-lease `asyncio.Lock`.

```bash
# Boot the pool server
uvicorn coliseum.server:app --host 0.0.0.0 --port 8100

# Drive it from a trainer
export COLISEUM_BASE_URL=http://127.0.0.1:8100
```

The contract shape is the standard lease-pool pattern used by every parallel-rollout RL framework. See [`coliseum/README.md`](coliseum/README.md) for the full env-var table and the migration map from the previous `openclaw_integration` package name.

---

## The HF Space UI

The Gradio app at [`huggingface.co/spaces/Madhav189/sre-env`](https://huggingface.co/spaces/Madhav189/sre-env) is mounted at `/` of the same uvicorn process that serves `/reset` + `/step`. Three tabs: Triage (live HTTP), Strategy (chained-episode runner), Operations (graph simulator). Every tab streams per-tick action, env response, reward delta, and the 5-component breakdown — `out=… valid=… fmt=… anti=… eff=…`.

Provider auth is whatever the user pastes (HF token, Anthropic key, OpenAI key, Together / Groq / Fireworks / DeepSeek base URL + key) plus the `OllamaProvider` for local-model runs. Tokens live only on the request instance — never logged, never persisted, never echoed in error messages.

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

Old tier names (`Tier.BASIC`, `Tier.ADVANCED`, `Tier.MAX`) are preserved as Enum aliases so existing callers keep working; importing them emits a `DeprecationWarning`. Same shape for `openclaw_integration` → [`coliseum`](coliseum/README.md).

---

## Tests + lint

```bash
make test            # 203 collected, all green at HEAD
ruff check .         # configured; pre-existing F401 cleanups tracked separately
openenv validate .   # green
```

Per-tier coverage breakdown:

| Suite | Count | What it covers |
|---|---|---|
| `unified_incident_env/tests/` | 62 | Triage env behaviour, baselines, procgen, ceiling bands |
| `tests/` | 141 | tier wrapper, runners, providers, UI router, MCP route |

The two CI invariants that keep the rubric calibrated:

- `test_heuristic_ceiling_is_in_band` — naive heuristic in `[0.65, 0.80]` on every template.
- `test_round2_baseline_resolves` — scripted-optimal `≥ 0.90` on the 6 round-2 templates.

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
│   ├─ ui/                    providers, router, policies, run.│
│   ├─ local.py               in-process CLI for Ollama models │
│   └─ exceptions.py          typed errors                     │
│                                                              │
│  coliseum/                  parallel-rollout pool server     │
│   ├─ server.py              FastAPI lease pool               │
│   └─ client.py              ArenaClient + create_arena_client│
│                                                              │
│  skill/                     Claude Code skill (Path A)       │
│   ├─ SKILL.md               agent instructions               │
│   ├─ tools/                 sre-gym HTTP client              │
│   └─ verified-runbooks/     12 per-template runbooks         │
└──────────────────────────────────────────────────────────────┘
```

Per-tier deep dives in [`docs/TRIAGE_TIER.md`](docs/TRIAGE_TIER.md) / [`docs/STRATEGY_TIER.md`](docs/STRATEGY_TIER.md) / [`docs/OPERATIONS_TIER.md`](docs/OPERATIONS_TIER.md). Reward design: [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md). Operator guide: [`execution.md`](execution.md). Architectural narrative: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Materials

- [`openenv.yaml`](openenv.yaml) — declares the three tiers, runnable kinds, scenario counts.
- [`pyproject.toml`](pyproject.toml) — Python package, deps, entry points.
- [`docs/`](docs/) — architecture, per-tier deep dives, reward design, scenario authoring guide, references.
- [`skill/`](skill/) — Claude Code skill packaging (Path A).
- [`coliseum/`](coliseum/) — parallel-rollout pool server.
- [`demo/`](demo/) — `run_demo.sh` end-to-end demo, `pitch.md` narrative.
- [`eval/`](eval/) — held-out split definition, results directory.
- [`train/data/`](train/data/) — teacher trajectories (Claude Opus + Llama-3.3-70B + scripted baselines).
- [`notebooks/`](notebooks/) — GRPO training (`01_basic_train_grpo_unsloth.ipynb`) and eval comparison (`02_basic_eval_comparison.ipynb`).

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, India 2026 — by the dakshdoesdev / Madhav189 team.
