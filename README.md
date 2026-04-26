---
title: SystemTruth
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# SystemTruth — a tier-escalating SRE training environment

> **Hackathon submission — OpenEnv-class, India 2026**
>
> - 📖 **Blog:** [BLOG.md](BLOG.md)
> - 🚀 **Live HF Space:** https://huggingface.co/spaces/Madhav189/SystemTruth
> - 💻 **GitHub:** https://github.com/Madhav-GPT/SystemTruth
> - 🧪 **Training notebook (Colab):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Re9pwkabEP4Cearc2hCMMqGdEjSSUjGu?usp=sharing) — same as [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb)
> - 📊 **Eval results:** [`eval/results/`](eval/results/)
> - 📜 **License:** Apache 2.0

**Each tier escalates a different dimension. Triage escalates compute, Strategy escalates horizon, Operations escalates realism.** That single sentence is the load-bearing claim of the project.

---

## What's in the box (the USP — read this first)

SystemTruth is **one runnable RL environment with three personas baked into it**. The same 11-action contract, the same 5-component reward rubric, the same termination shape — escalated along three orthogonal axes that map to the three real bottlenecks SRE-agent training loops actually hit.

![SystemTruth architecture — three tiers under one shared 11-action interface + 5-component rubric](docs/blog/system_architecture.png)

### One environment, three tiers, three different bottlenecks

| Tier | Bottleneck | Persona | What it teaches |
|---|---|---|---|
| **Triage** | **Compute** | ML student / Kaggle, $30 of HF credits | causal mapping under tight context — pre-digested observations, dense reward shaping, 8K context, 11-action space, 8–13 ticks per episode |
| **Strategy** | **Horizon** | Seed-stage startup, $300–500 budget | long-horizon planning across chained incidents — multi-incident chains with persistent state, unresolved alerts and pending deploys carry forward, 60–90 ticks |
| **Operations** | **Realism** | Enterprise SRE platform, 8×A100/H100 cluster | authentic tool use against irreversible actions — 22-node service graph, 11 chaos patterns pinned to real production post-mortems, 110–180+ actions per episode |

The escalation axis is the entire pitch. Most RL environments stratify by *difficulty* (more scenarios, longer episodes, harder rewards). SystemTruth stratifies by **the dimension that actually limits the training loop for that persona**:

- A junior on-call learning to triage faces a different problem (cognitive efficiency under tight context) than a senior SRE running a multi-incident postmortem (state tracking across long horizons), which faces a different problem from an enterprise platform team operating against an actively chaos-engineered cluster (irreversible actions, partial observability, real wall-clock).
- Their training signals, episode shapes, observation richness, and reward structures should not look the same.
- SystemTruth takes that observation seriously and stratifies its tiers along *the dimension that actually limits the persona's training loop*.

### The shared 11-action contract

Every tier — Triage, Strategy, Operations — speaks the same eleven Pydantic-validated actions. **One contract, three escalation envelopes:**

```
query_logs(service)            query_metrics(service, metric)
query_dependencies(service)    query_deploys(service)
rollback_deploy(service)       restart_service(service)
isolate_service(service)       run_check(check_name)
submit_hypothesis(hypothesis)  escalate
declare_resolved
```

A successful episode is `gather evidence → submit_hypothesis → rollback_deploy → restart_service → both run_check pass → declare_resolved`. Wrong rollback target, premature restart, or premature resolution all return negative reward and a typed `failure_type`. The contract refuses to be gamed.

### The episode lifecycle, illustrated

The lifecycle below is the Triage tier in detail; Strategy chains N of them with horizon-decay, Operations runs one of them inside a graph-mutation simulator. **The shape is shared across all three tiers** — the simulator under it is what changes.

![SystemTruth episode lifecycle — Triage tier, same shape inherited by Strategy and Operations](docs/blog/episode_lifecycle.png)

Eleven numbered stages, each producing a measurable signal:

1. **`reset(scenario_id)`** — env emits the initial observation: tick counter, workflow stage, incident summary, active alerts, noise alerts (decoys), service health (cpu/mem/err/latency), user impact, SLO burn rate, checks, allowed actions.
2. **Evidence gathering loop** — agent calls `query_logs / query_metrics / query_dependencies / query_deploys`. After every step the env computes a per-tick **shaped reward** as a potential difference (`Δ critical_service_health × 0.55 + Δ (1 − user_impact) × 0.20 + Δ (1 − slo_burn_rate) × 0.15 + containment_applied × 0.10`) minus `step_cost`, plus `bonus`, minus `penalty`.
3. **`submit_hypothesis(root_cause, affected_services, confidence, recommended_next_action)`** — the world-modelling primitive. Confidence is a `float ∈ [0,1]` the agent must commit to.
4. **Hypothesis correctness check** — if the root cause matches truth, the agent gets an in-episode bonus up to ~0.12 (idempotent — second identical hypothesis scores 0). If wrong, the agent loops back to investigation with a new observation.
5. **`rollback_deploy(service)`** — the irreversible action. Wrong target = `unsafe_action_penalty` (0.08 medium / 0.12 hard). Correct target sets `cause_removed = True` and unblocks restart.
6. **`restart_service(service)`** — only valid if scenario requires it. Guard: if cause not removed, premature-restart penalty fires and state re-inherits the bad config.
7. **`run_check("end_to_end" | "database_recovery")`** — verification gate. If checks fail, agent loops back to investigation.
8. **`declare_resolved`** — terminal action. Guard: if checks not passed, `premature_resolution_penalty` (0.20 / 0.30) fires.
9. **Episode terminates** — terminal state emitted.
10. **Compute composite from terminal state** — the 5-component rubric below evaluates outcome / action_validity / format / anticheat / efficiency, sums to 1.0 with weighted clamping to `[0.01, 0.99]`.
11. **Reference scores anchor the rubric** — random `0.417` (0/36 resolved), naive heuristic `0.749` (0/12 resolved), scripted-optimal `0.938` (12/12 resolved). The 0.20 gap from `0.80 → 1.00` is what GRPO trains into.

**Cross-tier extension:**
- **Strategy** chains N Triage episodes, applies a `horizon_decay_factor × mean(per-phase composite)` to the final reward.
- **Operations** runs the same lifecycle inside a graph-mutation simulator over a 22-node service topology, same rubric, same horizon-decay weighting.

### Reward rubric — the engineering crown jewel

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

Each component defends against a specific cheat:

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

The `confidence ∈ [0,1]` field is part of the structured `HypothesisPayload` Pydantic model the agent emits; the calibration sub-term reads it directly. **A model that bluffs high confidence on a wrong root cause is worse than one that hedges.** That's the world-modelling primitive — the env grades the agent's belief, not just its prediction.

Two CI invariants pin the rubric in place on every commit:
- **Heuristic ceiling `[0.65, 0.80]`** — `test_heuristic_ceiling_is_in_band` enforces this band on every template. The 0.20 gap from 0.80 → 1.00 is the GRPO training target.
- **Scripted-expert floor `≥0.90`** — `test_round2_baseline_resolves` enforces ≥0.90 on the round-2 templates.

Tighten either side and the gradient signal collapses; loosen either side and a memorising model can game the rubric. The band is the load-bearing engineering claim.

---

## Coliseum — parallel-rollout pool server

[`coliseum/`](coliseum/) wraps the Triage env in a lease-based HTTP contract so a GRPO trainer's parallel-rollout side can drive the env without holding an in-process Python instance per worker:

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

The environment is the project. Training scripts orbit around it. We ran a real end-to-end Triage run on the day of the deadline — pipeline works, results are below the heuristic plateau, the gap is real, and we're saying so.

### What we ran

Pipeline lives in [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb) — also openable in Colab via the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Re9pwkabEP4Cearc2hCMMqGdEjSSUjGu?usp=sharing). Target A100 80GB, ~2-3h end-to-end:

1. **SFT cold-start** — Unsloth + Qwen2.5-7B-Instruct (4-bit) + LoRA r=32, 50 steps × batch 16 on a 999-example diverse trajectory corpus. Eval perplexity 1.755 (healthy `[1.5, 3.0]` band). Saved to `outputs/qwen25_7b_sft_final/`.
2. **GRPO online** — TRL's `GRPOTrainer`, 40 steps × K=2 rollouts. Reward = composite + scenario-aware first-action bonus. Saved to `outputs/qwen25_7b_grpo_final/`.
3. **Held-out eval** — 12 `__p05` scenarios × 3 seeds = 36 episodes per policy, 5 policies compared.

### What it produced

![SystemTruth Triage holdout eval — Qwen2.5-7B, 12 scenarios × 3 seeds](eval/results/qwen25_7b_comparison_hero.png)

| policy | mean | median | p25 | p75 | resolved_rate |
|---|---|---|---|---|---|
| random | 0.342 | 0.378 | 0.340 | 0.380 | 0/36 |
| **qwen25-7b-sft-only** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| **qwen25-7b-grpo** | **0.379** | 0.380 | 0.378 | 0.380 | 0/36 |
| heuristic (queries + correct hypothesis, no remediation) | 0.704 | 0.705 | 0.703 | 0.705 | 0/36 |
| scripted-optimal | 0.938 | 0.939 | 0.937 | 0.940 | 36/36 |

![Per-template mean score by policy](eval/results/qwen25_7b_comparison_per_template.png)

Honest reading:
- SFT lifted the model 11% above random (0.342 → 0.379). Format-learning worked: the trained model produces 100% schema-valid `action_type` JSON.
- GRPO did not move the mean further on K=2 / 40 steps / 7B. The advantage signal exists but the budget is too small to overcome the heuristic plateau at 0.704.
- The env's rubric refuses to give partial credit for "looks right" output. A model that emits a polished hypothesis but never calls `rollback_deploy` is rewarded for zero remediation, and the 0.45 `outcome` slice stays empty. **The numbers don't lie about the work the model didn't do** — and that's exactly the rubric's job.

### What's bottlenecking us

The training-time weak spot is **data scale and step budget**, not the env. The 120-episode corpus + 50-step SFT + 40-step GRPO is what fits inside a hackathon weekend on one A100. The env, the rubric, and Coliseum are all sized for the real run; the trajectory corpus and the GRPO budget are not. Scaling either side (more teacher trajectories, more GRPO steps with K=4 rollouts, longer eval window) is the obvious next move and the bottleneck stops being the env.

The training scripts in [`train/`](train/) are working as written for the dataset we have — `build_corpus.py` produces a clean 60/20/20 quality-stratified split, `eval_sweep.py` drives the held-out comparison, `run_expert_collection.py` harvests teacher trajectories. They don't need rewriting; they need more data flowing through them.

---

## Quickstart

### 5-minute local demo (no API keys, no server, no GPU)

```bash
pip install -e .
ollama pull llama3.2
python -m sre_gym.local triage worker_deploy_cascade
```

The CLI drives `UnifiedIncidentEnvironment` directly and prints per-tick reward, the 5-component score breakdown, and a final summary.

### Live HF Space (Triage tier, hosted)

Open https://huggingface.co/spaces/Madhav189/SystemTruth. Pick a tier, paste an HF token, click **▶ run eval**. Each tick streams the action, env response, reward delta, and the 5-component breakdown.

### Local server + Gradio UI

```bash
make install
make dev                                              # FastAPI + Gradio on :7860
python -m sre_gym.strategy run cascading_release_train
python -m sre_gym.operations run ecommerce_vibecoded_saas --chaos rls_silent_leak
```

The FastAPI server speaks the OpenEnv contract (`/reset /step /state /tasks /baseline /grader /status /health /metadata /schema`) plus an MCP JSON-RPC route at `/mcp`.

---

## Two-paths agent design

The repo ships **two independent paths** to a working SRE agent. They share the env contract but trade compute for capability differently.

### Path A — verified-runbook skill (zero training)

[`skill/`](skill/) packages the env as a Claude Code skill. The agent reads scenario evidence, writes a verified runbook on a clean solve, and reads the runbook on the next attempt. No training, no GPU.

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
bash demo/run_demo.sh                                # end-to-end demo
```

### Path B — GRPO-trained adapter (one A100, ~2–3h on 7B)

The training pipeline above. Path A is what an agent ships *today*; Path B is what raises the floor on the templates it sees over and over.

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
ruff check .
openenv validate .   # green
```

The two CI invariants that keep the rubric calibrated:

- `test_heuristic_ceiling_is_in_band` — naive heuristic in `[0.65, 0.80]` on every template.
- `test_round2_baseline_resolves` — scripted-optimal `≥ 0.90` on the round-2 templates.

---

## Materials

- [`BLOG.md`](BLOG.md) — the hackathon blog (with all 6 assets in `docs/blog/`)
- [`openenv.yaml`](openenv.yaml) — declares the three tiers, runnable kinds, scenario counts
- [`docs/`](docs/) — architecture, per-tier deep dives, reward design, scenario authoring
- [`docs/blog/`](docs/blog/) — visuals: lifecycle, architecture, hero, topology, rubric donut, chaos timeline, two-paths, baselines bar
- [`skill/`](skill/) — Claude Code skill packaging (Path A)
- [`coliseum/`](coliseum/) — parallel-rollout pool server
- [`demo/`](demo/) — `run_demo.sh` end-to-end demo, `pitch.md` narrative
- [`eval/`](eval/) — held-out split definition, results directory with the latest eval CSV + plots
- [`train/data/`](train/data/) — teacher trajectories (Claude Opus + Llama-3.3-70B + scripted baselines + 120-episode v2 corpus)
- [`notebooks/`](notebooks/) — Triage SFT→GRPO training (`01_triage_train_grpo_qwen25_7b.ipynb`), eval comparison (`02_triage_eval_compare_all.ipynb`), Strategy + Operations walkthroughs

---

## License

Apache 2.0. Built for the OpenEnv-class hackathon, India 2026 — by the Madhav-GPT / dakshdoesdev team.
