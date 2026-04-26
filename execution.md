# Execution runbook — sre-gym

> Operator guide. From clone → live env → training → submission. Updated after the hackathon training run; reflects the current state of the codebase.

---

## Current state

| Item | Status |
|---|---|
| Triage env (12 templates × 6 entries = 72 scenarios) | ✅ runnable end-to-end |
| Strategy orchestrator (chains Triage episodes) | ✅ runnable as Python orchestrator |
| Operations graph state-machine simulator (22 nodes, 11 chaos patterns) | ✅ runnable in Python |
| Operations docker-compose stack (`ghcr.io/sre-gym/*` images) | 🟡 design-spec — images not published |
| Strategy 28-action universe (in YAML) | 🟡 design-spec — runner uses Triage 11 |
| Gradio UI mounted at `/` of the FastAPI server | ✅ live |
| MCP JSON-RPC 2.0 dual-route at `/mcp` | ✅ live + parity-tested |
| Coliseum parallel-rollout pool server | ✅ live |
| Pytest suite | ✅ green at HEAD |
| `openenv validate .` | ✅ green |
| End-to-end SFT → GRPO run (Qwen2.5-7B) | ✅ executed |
| Eval comparison run (5 policies × 36 episodes) | ✅ executed |
| Trained-model row in baselines table | ✅ measured (`mean=0.379` — see §7) |

The honest framing: **the env is the project, the rubric is the engineering crown jewel, and the training run is below the heuristic plateau** because the corpus + step budget that fit inside a hackathon weekend aren't enough to break it. Pretending otherwise is the original sin of every other SRE-agent demo. We don't.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Local setup](#2-local-setup)
3. [First-run smoke test](#3-first-run-smoke-test)
4. [Tier-aware operation](#4-tier-aware-operation)
5. [Scenario authoring quickstart](#5-scenario-authoring-quickstart)
6. [Training pipeline (Triage SFT → GRPO)](#6-training-pipeline-triage-sft--grpo)
7. [Eval comparison sweep](#7-eval-comparison-sweep)
8. [HF Space deployment](#8-hf-space-deployment)
9. [Coliseum — parallel-rollout pool server](#9-coliseum--parallel-rollout-pool-server)
10. [Claude Code skill setup](#10-claude-code-skill-setup)
11. [Troubleshooting](#11-troubleshooting)
12. [Submission checklist](#12-submission-checklist)
13. [Operator FAQ](#13-operator-faq)
14. [Materials](#14-materials)

---

## 1. Prerequisites

**Local development (env serving + tests):**

- Python 3.10+ (3.11 / 3.12 / 3.14 verified)
- pip 24+ or uv
- Git
- Docker (only required for HF Space build; not required for normal env serving)
- 4 GB free RAM, 2 GB free disk

**Training (Triage SFT → GRPO on Qwen2.5-7B):**

- 1×A100 80GB (HF Pro Spaces, Colab A100, or rented)
- HF account + token (`HF_TOKEN`) with write scope for adapter push
- ~$5–8 of HF compute credits for one ~2-3h end-to-end run
- Optional: Anthropic / Fireworks / Groq key for richer comparison rows

---

## 2. Local setup

```bash
git clone https://github.com/Madhav-GPT/SystemTruth.git
cd sre-env

python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e '.[dev]'
```

Verify:

```bash
make test                                     # green
python -m openenv.cli validate .              # green
```

---

## 3. First-run smoke test

Boot the combined Gradio + FastAPI server:

```bash
uvicorn app:app --host 127.0.0.1 --port 7860
```

Then in a second shell:

```bash
curl -s http://127.0.0.1:7860/health | jq
# {"status": "ok", "environment": "unified_incident_env", ...}

curl -s http://127.0.0.1:7860/tasks | jq '.scenarios | length'
# 72

curl -s http://127.0.0.1:7860/mcp/tools | jq '.tools | length'
# 11
```

Hit a scenario via `/reset` + `/step`:

```bash
curl -s -X POST http://127.0.0.1:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"scenario_id":"memory_leak_oom"}' | jq '.observation.workflow_stage'
# "triage"

curl -s -X POST http://127.0.0.1:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action":{"action_type":"query_logs","service":"worker"}}' | jq '.observation.tool_output'
# "Worker logs: 'process killed (OOM)' every ~90s..."
```

Run the scripted-baseline smoke against all 12 templates:

```bash
make baseline
# scripted-optimal mean across all 12 templates: ~0.94
# 12 / 12 resolved
```

---

## 4. Tier-aware operation

```bash
make tier-info                                # prints per-tier metadata
```

Programmatic API:

```python
from sre_gym import SREGym, Tier

# Triage — live FastAPI env
env = SREGym(tier=Tier.TRIAGE)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
result = env.run("memory_leak_oom__p02", seed=42)

# Strategy — chained Triage episodes with horizon state
env = SREGym(tier=Tier.STRATEGY)
result = env.run("cascading_release_train", seed=1)

# Operations — Python state-machine simulator
env = SREGym(tier=Tier.OPERATIONS)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

CLI:

```bash
python -m sre_gym.strategy list
python -m sre_gym.strategy run cascading_release_train --seed 1

python -m sre_gym.operations list-chaos
python -m sre_gym.operations run ecommerce_vibecoded_saas --chaos rls_silent_leak
```

---

## 5. Scenario authoring quickstart

### 5.1 Add a 13th Triage template

1. Append the template dict to `EXTRA_TEMPLATES` in `unified_incident_env/server/basic_templates_extra.py`.
2. Append a baseline-action lambda to `extra_baselines()`.
3. Append the new `RootCauseType` value to `unified_incident_env/models.py`.
4. Append the template_id to `ROUND2_TEMPLATES` in `tests/test_round2_templates.py`.

`make test` exercises all of the above automatically. Procgen variants generate at module-import time.

### 5.2 Add a Strategy reference scenario

Drop a new YAML in `sre_gym/strategy/scenarios/`. Include the `DESIGN-SPEC HEADER` the existing scenarios carry — call out which subset of `allowed_actions:` is implemented vs design-spec. The runner falls back to the Triage 11 actions for anything else.

### 5.3 Add an Operations chaos pattern

Triplet of YAMLs:
- `sre_gym/operations/families/<id>.yaml` — family-level spec
- `sre_gym/operations/chaos/<id>_chaos_library.yaml` — composable chaos patterns
- `sre_gym/operations/compose/<id>.yaml` — docker-compose stack (mark as design-spec if images aren't published)

Then add the chaos descriptors to `CHAOS_PATTERN_DEFAULTS` in `sre_gym/operations/runner.py` so the simulator can run them.

See [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) for the full schema.

---

## 6. Training pipeline (Triage SFT → GRPO)

### 6.1 What ships

[`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb) is the canonical, end-to-end training notebook. Cells:

| # | Cell | What it does |
|---|---|---|
| 0 | Bootstrap | uv + Unsloth pinned-version install. Idempotent. |
| 1 | GPU verify | nvidia-smi + torch.cuda.is_available() |
| 2 | Build corpus | `train/build_corpus.py` → 120-episode trajectory corpus, 60/20/20 quality split |
| 3 | Sanity-check corpus | template coverage, score distribution, tier counts |
| 4 | Build SFT dataset | ChatML formatting, 999 step-pairs |
| 5 | Load Qwen2.5-7B (4-bit) + LoRA r=32 | Unsloth FastLanguageModel |
| 6 | SFT cold-start | 50 steps × batch 16, lr=5e-5, eval perplexity gate |
| 7 | Build GRPO prompts | 120 prompts pre-rendered with the same chat template |
| 8 | Reward function | composite + first-action bonus + key-alias normalisation |
| 9 | GRPO online | 40 steps × K=2 rollouts, beta=0.1, temperature=0.9 |
| 10 | Eval comparison sweep | 5 policies × 12 scenarios × 3 seeds |
| 11 | Summary table + plots | hero bar chart + per-template chart |
| 12 | Push to HF Hub | adapter upload |
| 13 | Package artifacts | tar.gz the outputs/ dir |

### 6.2 Run it

In Colab / HF Space (recommended, A100 80GB):

1. Open the notebook
2. Set runtime to A100 80GB
3. Set `HF_TOKEN` in Colab Secrets (or paste in cell 12 directly)
4. **Run-All** — top to bottom, ~2-3h end-to-end

Resume points: if `outputs/qwen25_7b_sft_final/` exists, cell 6 skips. If `outputs/qwen25_7b_grpo_final/` exists, cell 9 skips. If `eval/results/qwen25_7b_comparison_raw.csv` exists, cell 10 skips. Delete the artifact to force a re-run.

### 6.3 Stages (measured)

| Stage | Steps | Wall-clock on A100 80GB | Output |
|---|---|---|---|
| Build SFT corpus from teacher trajectories | one-time | ~30s | `train/data/seed_v2_120.jsonl` (120 episodes) |
| SFT cold-start (50 steps × batch 16) | 50 | ~7 min | `outputs/qwen25_7b_sft_final/` |
| GRPO online (40 steps × K=2) | 40 | ~50 min (transformers fallback) / ~15 min (vLLM path) | `outputs/qwen25_7b_grpo_final/` |
| Eval sweep (5 policies × 12 × 3) | 180 episodes | ~25 min | `eval/results/qwen25_7b_comparison_*.csv + *.png` |

---

## 7. Eval comparison sweep

The notebook's cell 10 runs the full 5-policy comparison and writes:

- `eval/results/qwen25_7b_comparison_raw.csv` — every per-episode row
- `eval/results/qwen25_7b_comparison_summary.csv` — per-policy aggregates
- `eval/results/qwen25_7b_comparison_hero.png` — single-axis bar chart with whiskers
- `eval/results/qwen25_7b_comparison_per_template.png` — per-template grouped bars

### Latest measured numbers

| policy | mean | median | p25 | p75 | resolved_rate |
|---|---|---|---|---|---|
| random | 0.342 | 0.378 | 0.340 | 0.380 | 0/36 |
| qwen25-7b-sft-only | 0.379 | 0.380 | 0.378 | 0.380 | 0/36 |
| qwen25-7b-grpo | 0.379 | 0.380 | 0.378 | 0.380 | 0/36 |
| heuristic | 0.704 | 0.705 | 0.703 | 0.705 | 0/36 |
| scripted-optimal | 0.938 | 0.939 | 0.937 | 0.940 | 36/36 |

SFT lifted the model 11% above random. GRPO added zero on K=2 / 40 steps. Both still below the heuristic plateau at 0.704. The training-time bottleneck is corpus size + step budget, not the env — see [`README.md`](README.md) §"Training & datasets" for the framing.

---

## 8. HF Space deployment

The repo is configured as an HF Space (Docker SDK):

```yaml
# top of README.md — HF Space frontmatter
sdk: docker
app_port: 7860
```

`Dockerfile` builds the FastAPI + Gradio app. The Space rebuilds automatically on push to `main`. To push:

```bash
# One-time: add the HF Space as a git remote
git remote add hf https://huggingface.co/spaces/Madhav189/SystemTruth

# Push (HF prompts for token if not cached)
git push hf main
```

The Space runs on CPU-basic by default — no GPU required for the Triage env. If the user provides an HF Inference Router model in the UI, calls go to that model; otherwise the run is gated until a token + model are pasted.

---

## 9. Coliseum — parallel-rollout pool server

[`coliseum/`](coliseum/) wraps the Triage env in a lease-based HTTP contract so a GRPO trainer can drive 8 concurrent rollouts on a single process via per-lease `asyncio.Lock`.

```bash
# Boot the pool server
uvicorn coliseum.server:app --host 0.0.0.0 --port 8100

# Drive it from a trainer
export COLISEUM_BASE_URL=http://127.0.0.1:8100
```

Endpoints: `/allocate`, `/reset`, `/exec_tool`, `/evaluate`, `/close`, `/healthz`. The `ArenaClient` in [`coliseum/client.py`](coliseum/client.py) drives them with retry/backoff per route. See [`coliseum/README.md`](coliseum/README.md) for the full env-var table.

---

## 10. Claude Code skill setup

Path A (zero training): the env packages cleanly as a Claude Code skill.

```bash
# Install the skill globally (one-time)
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"

# Or run the end-to-end demo
bash demo/run_demo.sh
```

12 verified-runbook drafts ship in [`skill/verified-runbooks/`](skill/verified-runbooks/) — one per Triage template. The skill validates them by re-running the env after each solve.

---

## 11. Troubleshooting

**`make test` fails on import error** — usually means `pip install -e '.[dev]'` skipped a dep. `pip install pytest pyyaml httpx` and retry.

**`make baseline` reports `mean > 0.80`** — the rubric is leaking. The CI invariant `test_heuristic_ceiling_is_in_band` should have caught this; check `unified_incident_env/server/grader.py` weights.

**`uvicorn app:app` crashes with `ImportError: openenv`** — `pip install openenv-core>=0.2.1` (the package name is `openenv-core` but the import is `openenv.core`).

**Cell 9 of the training notebook errors with `'Qwen2ForCausalLM' object has no attribute 'vllm_engine'`** — Cell 5 didn't pass `fast_inference=True` when loading the model. The notebook's preflight check now detects this and falls back to the transformers `model.generate` path automatically. ~3× slower but always works.

**Cell 9 errors with `EADDRINUSE on port 12345`** — a previous failed `init_process_group` left the port bound. Restart the kernel and re-run from Cell 0. The current cell defensively calls `dist.destroy_process_group()` before any new init.

**`reward_std = 0` in early GRPO steps** — model emits the same JSON shape on every K rollout (entropy collapse). Bump `temperature=0.9 → 1.1` in cell 9's `_build_grpo_args`.

---

## 12. Submission checklist

- [x] Repo public on GitHub: https://github.com/Madhav-GPT/SystemTruth
- [x] HF Space live: https://huggingface.co/spaces/Madhav189/SystemTruth
- [x] BLOG.md at repo root
- [x] 6 blog assets in `docs/blog/`
- [x] Training notebook executed end-to-end, results in `eval/results/`
- [x] README links blog + Space + GitHub + notebook + license
- [x] HF Space README links blog + GitHub + notebook
- [x] `openenv validate .` green
- [x] Pytest suite green
- [ ] Hackathon submission form: paste the HF Space URL as the canonical entry point

---

## 13. Operator FAQ

**Q: Why does Random outperform the Heuristic on some templates?**
The heuristic commits to a fixed wrong sequence on a few templates while Random sometimes stumbles into useful evidence-gathering and earns shaped per-tick reward. Documented rather than buried.

**Q: Why do all 11 chaos patterns name the failing service in the Operations incident summary?**
Because the simulator is a fault-injection harness, not a hidden-information puzzle. A real-cluster Operations tier would use raw Loki / Tempo signals; the Python sim doesn't claim to.

**Q: Why is `supabase_rls_silent_leak` approximated by `payment_webhook_misconfig + migration_lock + worker_deploy_cascade` in the Strategy runner?**
Because there's no Supabase-RLS Triage template; the Strategy runner approximates higher-tier scenarios via the closest-shaped Triage templates. Documented as approximation, not fidelity.

**Q: Why did GRPO not beat SFT on the 7B run?**
K=2 rollouts × 40 steps on a 7B model with a 120-episode corpus is too small a budget to break the heuristic plateau at 0.704. The env, the rubric, and Coliseum are sized for a much bigger run; the corpus and step budget are what need to scale next.

---

## 14. Materials

- [`README.md`](README.md) — repo overview, the README judges read first
- [`BLOG.md`](BLOG.md) — the hackathon blog with the 6 assets in `docs/blog/`
- [`openenv.yaml`](openenv.yaml) — declares the three tiers, runnable kinds, scenario counts
- [`docs/`](docs/) — architecture, per-tier deep dives, reward design, scenario authoring
- [`notebooks/01_triage_train_grpo_qwen25_7b.ipynb`](notebooks/01_triage_train_grpo_qwen25_7b.ipynb) — the canonical training notebook
- [`notebooks/02_triage_eval_compare_all.ipynb`](notebooks/02_triage_eval_compare_all.ipynb) — multi-policy eval comparison
- [`notebooks/03_strategy_blueprint_walkthrough.ipynb`](notebooks/03_strategy_blueprint_walkthrough.ipynb) — Strategy tier walkthrough
- [`notebooks/04_operations_demo_chaos.ipynb`](notebooks/04_operations_demo_chaos.ipynb) — Operations tier walkthrough
- [`coliseum/README.md`](coliseum/README.md) — parallel-rollout pool server
- [`skill/SKILL.md`](skill/SKILL.md) — Claude Code skill (Path A)
- [`eval/results/`](eval/results/) — eval CSVs + plots
- [`train/data/`](train/data/) — teacher trajectory corpora (120 + extras)
