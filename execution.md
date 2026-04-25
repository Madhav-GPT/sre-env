# Execution runbook — sre-gym v3.0

> The full operator guide. From clone to trained model to HF Space deploy. Every command in this doc has been verified end-to-end on macOS + Linux as of April 25, 2026.

---

## Table of contents

1. [Prerequisites + system requirements](#1-prerequisites--system-requirements)
2. [Local setup (5 minutes)](#2-local-setup-5-minutes)
3. [First-run smoke test (5 minutes)](#3-first-run-smoke-test-5-minutes)
4. [Tier-aware operation](#4-tier-aware-operation)
5. [Scenario authoring quickstart](#5-scenario-authoring-quickstart)
6. [Running the Basic-tier training pipeline](#6-running-the-basic-tier-training-pipeline)
7. [Running the eval comparison sweep](#7-running-the-eval-comparison-sweep)
8. [Building the seed dataset (collecting teacher trajectories)](#8-building-the-seed-dataset-collecting-teacher-trajectories)
9. [HF Space deployment](#9-hf-space-deployment)
10. [Async GRPO via OpenClaw-RL](#10-async-grpo-via-openclaw-rl)
11. [Claude Code skill setup](#11-claude-code-skill-setup)
12. [Performance + cost reference](#12-performance--cost-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Submission-day checklist](#14-submission-day-checklist)
15. [FAQ for operators](#15-faq-for-operators)

---

## 1. Prerequisites + system requirements

**Local development (Basic tier env serving + tests):**

- Python 3.10+ (3.11 or 3.12 recommended; 3.14 verified)
- pip 24+ or uv
- Git
- Docker (only required for HF Space build + Max-tier compose; not required for normal env serving)
- 4 GB free RAM, 2 GB free disk

**Training (Basic tier, end-to-end GRPO):**

- 1×A100 40GB (HF Pro Spaces, Colab A100, or rented Lambda/Vast/CoreWeave)
- *Or* 1×L4 24GB (drops to Qwen2.5-1.5B; ~22h instead of ~12h)
- *Or* 1×H100 80GB (can run Qwen2.5-7B; ~6h)
- 80 GB scratch disk
- HF account + token (set `HF_TOKEN`) for adapter push
- Optional: Anthropic API key (Claude teacher trajectories), Fireworks/Groq API key (Llama-3.3-70B comparison)

**Max tier (operator only — not provisioned in this repo):**

- 8× A100/H100 cluster (on-prem or rented)
- ~$40–150/day cluster cost
- Sandboxed Stripe test creds, sandboxed Supabase project, sandboxed git remote
- Docker Compose v2.x or k3d
- ~$1–2k registry-cost commitment for stub-image hosting

---

## 2. Local setup (5 minutes)

### 2.1 Clone + install

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm.git
cd sre-enginnerllm

python3 -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e '.[dev]'
```

### 2.2 (Optional) install training extras

Only needed if you'll re-run notebook 01 locally rather than in Colab:

```bash
pip install -e '.[dev,train]'
# adds: datasets, transformers, peft, accelerate, trl, anthropic
# Unsloth itself pulls in via the notebook (bnb-4bit weights only fit on CUDA)
```

### 2.3 Verify install

```bash
make test                                       # 74 tests, ~2s
python -m openenv.cli validate .                # OpenEnv manifest check
```

Expected output:

```
74 passed in 2.0s
```

If you see import errors, the most common culprits are:

- `ModuleNotFoundError: openenv` — re-run `pip install -e '.[dev]'`
- `pydantic.ValidationError on UnifiedIncidentAction` — your scenarios JSON references an action_type that's not in the Literal; check `unified_incident_env/models.py:ActionType`

---

## 3. First-run smoke test (5 minutes)

### 3.1 Boot the env

```bash
python scripts/run_server.py --port 8000 &
# or:
make dev                                        # uvicorn with --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3.2 Hit the catalog routes

```bash
curl -s http://127.0.0.1:8000/health | jq
# {"status": "ok", "environment": "unified_incident_env", "version": "2.0.0", "stages": [...]}

curl -s http://127.0.0.1:8000/tasks | jq '.scenarios | length'
# 72

curl -s http://127.0.0.1:8000/tasks | jq '[.scenarios[].id] | sort | .[0:5]'
# ["auth_token_expiry", "auth_token_expiry__p01", "auth_token_expiry__p02", ...]
```

### 3.3 Run a single optimal episode against any template

```bash
curl -s -X POST http://127.0.0.1:8000/reset \
  -H 'content-type: application/json' \
  -d '{"scenario_id":"memory_leak_oom"}' | jq '.workflow_stage,.active_alerts[].message'

curl -s -X POST http://127.0.0.1:8000/step \
  -H 'content-type: application/json' \
  -d '{"action":{"action_type":"query_logs","service":"worker"}}' | jq '.tool_output'
# "Worker logs: 'process killed (OOM)' every ~90s. Memory growth pattern began..."
```

### 3.4 Run all 12 baselines as a smoke check

```bash
make baseline
# scripted-optimal mean across all 12 templates: ~0.750
# all 12 resolved
```

If the overall mean ever exceeds 0.80, the rubric is leaking — see `docs/REWARD_DESIGN.md` §4.

### 3.5 Run the full sweep + write JSONL

```bash
PYTHONPATH=. python scripts/eval_baseline.py \
  --output eval/results/scripted_baseline.jsonl

# Writes 72 rows (12 templates × 6 procgen variants each).
```

---

## 4. Tier-aware operation

### 4.1 Inspect any tier

```bash
make tier-info
```

Or programmatically:

```python
from sre_gym import SREGym, Tier

# Basic — runnable
env = SREGym(tier=Tier.BASIC)
print(env.describe())
# {tier: 'basic', escalation_dimension: 'compute', persona: '...', scenario_count: 72, ...}

obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})

# Advanced — design-only
env = SREGym(tier=Tier.ADVANCED)
for spec in env.list_scenarios():
    print(spec['id'], spec.get('difficulty'), spec.get('reference_trajectory_length'))
# cascading_release_train     hard       80
# observability_pipeline_outage hard     70
# supabase_rls_silent_leak    very_hard  60

try:
    env.reset()
except Exception as e:
    print(e)            # TierNotRunnableError pointing at docs/ADVANCED_TIER.md

# Max — design-only with 22-service compose
env = SREGym(tier=Tier.MAX)
fam = env.list_scenarios()[0]
print(f"{fam['id']}: {len(fam['topology']['services'])} services, {fam['scenario_population']['size']} instances")
```

### 4.2 Open the per-tier walkthroughs

Notebooks 03 and 04 walk through the design-only tiers without requiring training:

```bash
jupyter notebook notebooks/03_advanced_blueprint_walkthrough.ipynb
jupyter notebook notebooks/04_max_demo_chaos.ipynb
```

Each renders the YAML specs, prints incident chains / chaos patterns / topology tables, and demonstrates `TierNotRunnableError` graceful-degradation.

---

## 5. Scenario authoring quickstart

### 5.1 Add a 13th Basic template (60 minutes)

```python
# 1. In unified_incident_env/server/basic_templates_extra.py, append to EXTRA_TEMPLATES:
EXTRA_TEMPLATES["my_new_template"] = {
    "id": "my_new_template",
    "difficulty": "medium",
    "name": "My New Template",
    "description": "A 1-3 sentence summary of the incident shape.",
    "root_cause": "1 sentence root cause description.",
    "optimal_ticks": 10,
    "max_ticks": 12,
    "critical_service_weights": {"api-gateway": 0.3, "cache": 0.0, "database": 0.4, "worker": 0.3},
    "reward_config": _STD_REWARD,
    "initial_services": {...},
    "initial_alerts": [...],
    "logs": {...},
    "metrics": {...},
    "dependencies": {...},
    "deploy_history": {...},
    "checks": {"database_recovery": "...", "end_to_end": "..."},
    "truth": {"root_cause": "my_new_root_cause", "affected_services": [...], "best_next_action": "rollback_deploy"},
    "remediation_recipe": {"rollback_target": "worker", "restart_target": "database", ...},
    "post_rollback_services": {...},
    "post_restart_services":  {...},
    "post_isolate_services":  {...},
    "post_rollback_user_impact": 0.30,
    # ... etc
    "degraded_services": {...},
    "failure_messages": {...},
    "difficulty_knobs": {"noise_services": [...], "noise_alerts": [...], "noise_logs": {...}},
}

# 2. Add the baseline:
def extra_baselines() -> dict[str, list[Any]]:
    return {
        ...,  # existing 6
        "my_new_template": lambda: [
            _ba("query_logs", service="...", rationale="..."),
            ...,
            _ba("declare_resolved", rationale="..."),
        ],
    }

# 3. In unified_incident_env/models.py, append to RootCauseType:
RootCauseType = Literal[
    ...,                                # existing 12
    "my_new_root_cause",                # new
]

# 4. In unified_incident_env/tests/test_round2_templates.py, append to ROUND2_TEMPLATES:
ROUND2_TEMPLATES.append("my_new_template")
```

Run `make test` — the parametrized tests will exercise the new template automatically. Procgen variants generate at module-import time. The `/tasks` route exposes 78 scenarios (72 + 6 new).

### 5.2 Add an Advanced reference scenario

Drop a new YAML in `sre_gym/advanced/scenarios/`. Required sections per [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md): `id`, `tier: advanced`, `difficulty`, `name`, `description`, `topology`, `incident_chain`, `allowed_actions`, `reward_dimensions`, `reference_trajectory_length`, `optimal_ticks`, `max_ticks`, `reference_trace`, `oncall_peer`, `success_criteria`.

### 5.3 Add a Max scenario family

Triplet of YAMLs:

- `sre_gym/max/families/<family_id>.yaml` — family-level spec (topology, scenario_population, allowed_actions, reward_model, reference_instance, operator_notes)
- `sre_gym/max/chaos/<family_id>_chaos_library.yaml` — composable chaos patterns
- `sre_gym/max/compose/<family_id>.yaml` — docker-compose stack

See [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) for the full schema and discipline (cost notes, isolation requirements, composability constraints).

---

## 6. Running the Basic-tier training pipeline

### 6.1 In Colab (recommended, A100 free tier)

1. Open [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) in Colab.
2. Set runtime to A100 (Runtime → Change runtime type → A100 GPU).
3. Set `HF_TOKEN` in Colab Secrets (sidebar key icon → Add).
4. Optional: set `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY` for richer seed data.
5. Run-All. End-to-end takes ~12h; progress reports every 25 steps.

The notebook auto-detects GPU memory and selects:
- A100 40GB → Qwen2.5-3B-Instruct, LoRA r=64, K=4 GRPO rollouts
- L4 24GB → Qwen2.5-1.5B-Instruct, LoRA r=32, K=2
- ≤16GB → Qwen2.5-0.5B, LoRA r=16, K=2

Output: trained adapter pushed to `dakshdoesdev/sre-gym-qwen25-3b-grpo` (rename in cell 9).

### 6.2 Locally (e.g. on rented A100)

```bash
pip install -e '.[dev,train]'
pip install 'unsloth>=2025.1.0' 'unsloth_zoo>=2025.1.0' 'wandb>=0.18'

export HF_TOKEN="hf_xxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxx"     # optional (Claude teacher data)
export FIREWORKS_API_KEY="fw_xxxx"         # optional (Llama-3.3-70B comparison)

jupyter nbconvert --to notebook --execute --output /tmp/train_output.ipynb \
  notebooks/01_basic_train_grpo_unsloth.ipynb
```

Outputs land in:
- `/content/sft-out/lora/` — SFT adapter
- `/content/grpo-out/lora/` — final GRPO adapter
- `/content/eval-results/trained_qwen.jsonl` — held-out eval results

### 6.3 Stage 1 — seed dataset (~$15 of API spend, ~2h)

If you want to skip the notebook and just rebuild the seed dataset:

```bash
PYTHONPATH=. python -c "
from unified_incident_env.server.environment import UnifiedIncidentEnvironment
from unified_incident_env.server.challenge import list_baselines, list_scenarios
import json

scenarios = list_scenarios().scenarios
train_ids = [s.id for s in scenarios if not s.id.endswith('__p05')]
out = []
for sid in train_ids:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=sid)
    bl = list_baselines(scenario_id=sid).baselines[0]
    for step in bl.actions:
        out.append({'scenario_id': sid, 'prompt': obs.prompt_text, 'completion': step.action.model_dump_json()})
        obs = env.step(step.action)
        if obs.done: break

with open('train/data/sft_seed.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r)+'\\n')
print(f'wrote {len(out)} pairs')
"
```

### 6.4 Stage 2 — SFT cold start (~3h on A100)

The notebook handles this automatically; the standalone path uses TRL's `SFTTrainer` with the snippet in cell 6 of the notebook.

### 6.5 Stage 3 — GRPO online (~6h on A100)

The notebook's cell 7 wraps `GRPOTrainer` with a custom `episode_reward()` callback that boots a fresh `UnifiedIncidentEnvironment` per completion and reads back `obs.reward` (the shaped per-tick reward). Group-relative advantages are computed automatically by TRL.

Hyperparameters:

| Setting | Value | Why |
|---|---|---|
| `num_generations` (K) | 4 | Group size for advantages |
| `temperature` | 0.7 | Encourages exploration without divergence |
| `learning_rate` | 5e-6 | Conservative — adapter is post-SFT, big steps overfit |
| `max_completion_length` | 256 | Action JSON fits; longer would burn rollout budget |
| `beta` (KL) | 0.04 | Standard TRL default |
| `max_steps` | 800 | ~6h on A100; ~10h on L4 |

### 6.6 Stage 4 — eval (~30min)

The notebook's cell 8 runs 36 episodes (3 per held-out scenario × 12 scenarios) and writes JSONL.

For a fuller comparison sweep across 7 policies, use notebook 02 (next section).

---

## 7. Running the eval comparison sweep

### 7.1 The 7-policy comparison (~4h with API calls, ~10min for local-only)

Open [`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) and Run-All. Policies evaluated:

1. Random (uniform over allowed_actions)
2. Heuristic (deterministic if-else, no LLM)
3. Scripted-optimal baseline
4. Llama-3.3-70B-Instruct (Fireworks; Groq fallback)
5. Claude Haiku 4.5
6. Claude Sonnet 4.6
7. Trained Qwen2.5-3B (loaded from `/content/grpo-out/lora` or HF Hub)

API keys required for full sweep:

- `ANTHROPIC_API_KEY` (Haiku + Sonnet rows)
- `FIREWORKS_API_KEY` *or* `GROQ_API_KEY` (Llama row)
- `HF_TOKEN` (for trained adapter pull if not local)

Outputs (in `eval/results/`):

- `comparison_raw.csv` — every per-episode row
- `comparison_summary.csv` — per-policy aggregates (mean / median / p25 / p75 / resolved)
- `comparison_table.csv` — printable comparison table for the README
- `comparison_per_template.png` — per-template box-and-whisker reward distributions
- `comparison_hero.png` — single-axis bar chart, the README hero figure

### 7.2 Local-only sweep (skip API rows)

If you only want random + heuristic + scripted (no API costs):

```bash
PYTHONPATH=. python scripts/eval_baseline.py --output eval/results/baseline.jsonl
```

This runs the scripted baseline across all 72 scenarios and writes one JSONL row per episode. Then plot:

```bash
PYTHONPATH=. python scripts/plot_curves.py eval/results/baseline.jsonl \
  --x-field tick_count --y-field final_score \
  --group-by template_id --title "scripted-optimal per template" \
  --output eval/results/baseline_curve.png
```

### 7.3 Reproducing prior numbers

The baseline numbers in the README's frontier-baselines table came from `train/data/eval_sweep_baselines.jsonl`. To re-run that sweep against the live HF Space:

```bash
python train/eval_sweep.py \
  --env-url https://dakshdoesdev-sre-gym.hf.space \
  --scenarios all \
  --policies random,heuristic \
  --episodes-per-scenario 5 \
  --output train/data/eval_sweep_baselines_repro.jsonl
```

For LLM rows (Llama / Claude):

```bash
export GROQ_API_KEY=...
python train/eval_sweep.py \
  --env-url https://dakshdoesdev-sre-gym.hf.space \
  --scenarios all \
  --policies groq \
  --groq-model llama-3.3-70b-versatile \
  --episodes-per-scenario 3 \
  --output train/data/llama33_70b_groq_repro.jsonl
```

---

## 8. Building the seed dataset (collecting teacher trajectories)

### 8.1 Teacher options

| Driver | Model | Rate limits | Cost per ~200 trajectories |
|---|---|---|---|
| `--driver anthropic` | Claude Opus 4.7 | 5 req/min on tier-1 | ~$15 |
| `--driver anthropic` | Claude Sonnet 4.6 | 50 req/min | ~$8 |
| `--driver fireworks` | Llama-3.3-70B-Instruct | depends on plan | ~$3 |
| `--driver groq` | Llama-3.3-70B-Versatile | 14k req/day free | $0 |
| `--driver heuristic` | scripted | unlimited | $0 |

Recommended mix for variance: Claude Opus (clean, optimal paths) + Groq Llama (noisier, more failures). The variance is the training signal.

### 8.2 Run trajectory collection

```bash
export ANTHROPIC_API_KEY=...
python train/collect_trajectories.py \
  --env-url http://127.0.0.1:8000 \
  --scenarios all \
  --models claude-opus-4-7 \
  --episodes-per-model 100 \
  --parallelism 2 \
  --driver anthropic \
  --output train/data/claude_opus_100.jsonl
```

Or via Groq free tier:

```bash
export GROQ_API_KEY=...
python train/collect_trajectories.py \
  --env-url http://127.0.0.1:8000 \
  --scenarios all \
  --models llama-3.3-70b-versatile \
  --episodes-per-model 200 \
  --parallelism 3 \
  --driver groq \
  --output train/data/llama_groq_200.jsonl
```

### 8.3 Compile to SFT format

```bash
python train/compile_claude_seed.py \
  --inputs train/data/claude_opus_100.jsonl train/data/llama_groq_200.jsonl \
  --output train/data/seed_combined.jsonl
```

Notebook 01 picks up `train/data/seed_combined.jsonl` automatically if present and folds it into the SFT corpus.

---

## 9. HF Space deployment

### 9.1 First-time setup

The Space `dakshdoesdev/sre-gym` is configured as Docker SDK with the [`Dockerfile`](Dockerfile) at the repo root. You need:

- HF account with collaborator access to the Space
- `HF_TOKEN` with `write` scope

### 9.2 Deploy

```bash
# Authenticate the HF CLI
huggingface-cli login

# Deploy
bash deploy/push_to_hf.sh
```

The deploy script:

1. Builds the docker image locally to verify
2. Pushes the repo to the HF Space's git remote
3. The Space rebuilds automatically (~3-4min for cpu-basic)
4. Health check: `curl https://dakshdoesdev-sre-gym.hf.space/health`

### 9.3 Verify the live Space

```bash
curl -s https://dakshdoesdev-sre-gym.hf.space/health | jq
curl -s https://dakshdoesdev-sre-gym.hf.space/tasks | jq '.scenarios | length'
# 72
```

### 9.4 Run a baseline against the live Space

```bash
python train/eval_sweep.py \
  --env-url https://dakshdoesdev-sre-gym.hf.space \
  --scenarios all \
  --policies heuristic \
  --episodes-per-scenario 1 \
  --output /tmp/live_check.jsonl
```

---

## 10. Async GRPO via OpenClaw-RL

For training at scale (multiple A100s, distributed), the OpenClaw-RL pool-server shim wraps the env in a lease-based interface compatible with [Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL).

### 10.1 Boot the pool server

```bash
python -m uvicorn openclaw_integration.pool_server:app --port 8100
```

Endpoints:

```
POST /allocate    {task_key}                → {lease_id}
POST /reset       {lease_id, scenario_id}   → {observation}
POST /exec_tool   {lease_id, tool_call}     → {observation}
POST /evaluate    {lease_id}                → {score}
POST /close       {lease_id}                → {ok}
GET  /healthz                               → {ok, active_leases, scenarios}
```

`asyncio.Lock` per lease, TTL reaper for stale sessions, automatic cleanup on close. Mirrors the `OpenClaw-RL/terminal-rl/remote/pool_server.py` contract exactly.

### 10.2 Generate trajectories with OpenClaw

[`openclaw_integration/generate_with_sre.py`](openclaw_integration/generate_with_sre.py) is an import-patch wrapper for OpenClaw's `terminal-rl/generate.py`. Three-file shim, no edits to OpenClaw-RL internals required.

See `openclaw_integration/README.md` for full launch instructions.

---

## 11. Claude Code skill setup

The repo ships with a Claude Code skill at [`skill/`](skill/) that lets you solve scenarios interactively from inside Claude Code.

### 11.1 Install

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

### 11.2 Use

In Claude Code:

```
> Solve the network_partition__p03 scenario in sre-gym.
> List all 12 templates and explain what each one teaches.
> Run an interactive session against memory_leak_oom.
```

The skill drives the env via [`skill/tools/sre_gym_client.py`](skill/tools/sre_gym_client.py). On any clean solve (score > 0.85), it appends a fresh runbook to [`skill/verified-runbooks/`](skill/verified-runbooks/). The next session reads existing runbooks back as priors — this is the recursive-runbook-amplification loop.

### 11.3 Manual CLI use

```bash
export SRE_GYM_BASE_URL=http://127.0.0.1:8000

python skill/tools/sre_gym_client.py list
python skill/tools/sre_gym_client.py solve worker_deploy_cascade
python skill/tools/sre_gym_client.py interactive payment_webhook_misconfig
```

---

## 12. Performance + cost reference

### 12.1 Wall-clock budgets (Basic-tier full pipeline)

| Stage | A100 40GB | L4 24GB | H100 80GB |
|---|---|---|---|
| Seed dataset build (Claude teacher, 200 traj) | ~2h ($15 API) | ~2h ($15 API) | ~2h ($15 API) |
| SFT cold start (500 steps) | ~3h | ~6h | ~1.5h |
| GRPO online (800 steps) | ~6h | ~13h | ~3h |
| Eval sweep (36 episodes) | ~30min | ~45min | ~15min |
| **Total (training + eval)** | **~12h** | **~22h** | **~7h** |
| Model variant | Qwen2.5-3B | Qwen2.5-1.5B | Qwen2.5-7B |

### 12.2 HF credit consumption

| GPU class | HF Pro Spaces credits/h | Total for full pipeline |
|---|---|---|
| A100 40GB | 4.13 | ~50 credits |
| L4 24GB | 0.80 | ~18 credits |
| H100 80GB | 9.39 | ~66 credits |

The per-team budget is 3 × 30 = 90 credits. Running on L4 lets the entire team's allotment cover ~5 full training runs. A100 covers ~1.8 full runs per teammate.

### 12.3 API spend (optional, for richer comparison rows)

| API call | Cost |
|---|---|
| Claude Opus seed (200 trajectories × ~12 actions) | ~$15 |
| Claude Sonnet seed (same) | ~$8 |
| Claude Haiku eval row (36 episodes × ~12 actions) | ~$0.50 |
| Claude Sonnet eval row | ~$2 |
| Llama-3.3-70B Fireworks eval | ~$0.30 |
| Llama-3.3-70B Groq | $0 (free tier, 14k req/day) |
| OpenAI GPT-4 (if substituted for Claude Opus) | ~$25 |

---

## 13. Troubleshooting

### 13.1 `ModuleNotFoundError: openenv` on `pip install -e .`

```bash
pip install --upgrade 'openenv-core>=0.2.1'
```

### 13.2 `pydantic.ValidationError: ServiceName` on a new template

You added a service that's not in the `ServiceName` Literal. Edit [`unified_incident_env/models.py`](unified_incident_env/models.py) and append it to either the main service pool (4 services: api-gateway / cache / database / worker) or the noise-service pool.

### 13.3 GRPO loss diverges after step ~200

Three usual causes:

- **Learning rate too high.** Drop from `5e-6` to `2e-6`. Adapter is post-SFT; big steps overfit.
- **`beta` too low.** Bump KL coefficient from 0.04 to 0.08.
- **Reward function returning NaN.** If the env ever crashes mid-rollout (e.g. an unhandled action_type), `episode_reward()` should return `-0.05` not `NaN`. Check the parser fallback.

### 13.4 Notebook 01 OOM on Qwen2.5-3B

Drop to Qwen2.5-1.5B by setting `BASE_MODEL` manually before cell 3 runs. Or reduce `LORA_RANK` from 64 to 32.

### 13.5 HF Space build fails with "out of disk"

The cpu-basic Space has 16GB ephemeral. The Dockerfile prunes apt cache; if you've added Python deps that exceed 4GB total wheel size, prune optional deps from `pyproject.toml` or use slim variants.

### 13.6 `make baseline` reports overall mean > 0.80

The rubric is leaking. Check what changed in `unified_incident_env/server/grader.py` since last clean run. The CI invariant `test_baseline_ceiling_is_hardened_below_080` should have caught this — re-run `make test`.

### 13.7 Wrong-target rollback test fails for a new template

Your `remediation_recipe.rollback_target` and the wrong-target test's `wrong` candidate match. The test picks the first candidate in `["api-gateway", "cache", "database", "worker"]` that isn't the correct target — make sure your template's correct target isn't `api-gateway` *and* `api-gateway` isn't the only candidate.

### 13.8 Claude Code skill says "no env at SRE_GYM_BASE_URL"

```bash
export SRE_GYM_BASE_URL=http://127.0.0.1:8000
# or, for the live Space:
export SRE_GYM_BASE_URL=https://dakshdoesdev-sre-gym.hf.space
```

### 13.9 `git push` returns 403

You're authenticated as a user without collaborator access. Either:
- Have a maintainer add you as a collaborator (Settings → Collaborators on the GitHub repo)
- `gh auth login` as the maintainer
- Push from the maintainer's machine

---

## 14. Submission-day checklist

48h before:

- [ ] All 74 tests green (`make test`)
- [ ] `openenv validate .` green
- [ ] HF Space rebuild successful (`/health` returns 200)
- [ ] Notebook 01 runs end-to-end on A100 in a single 12h session
- [ ] Notebook 02 produces `eval/results/comparison_hero.png` with trained-Qwen above Haiku
- [ ] `train/data/seed_combined.jsonl` exists and has ≥ 200 rows
- [ ] Trained adapter pushed to `dakshdoesdev/sre-gym-qwen25-3b-grpo`

24h before:

- [ ] README links all check (HF Space, GitHub repo, docs files, notebook files, scripts)
- [ ] `eval/results/comparison_table.csv` matches the row counts in the README's frontier-baselines table
- [ ] `git status` is clean; `git log` shows the submission commit at HEAD
- [ ] HF Space has a recent successful build (no auto-rollback)

Submission window:

- [ ] Mini-blog or YouTube video link added to README §17 "materials linked"
- [ ] Submit form references the GitHub repo URL + HF Space URL
- [ ] `git tag submission-2026-04-26 && git push --tags`

---

## 15. FAQ for operators

**Q: I only have a free Colab T4. Can I train at all?**
A: Yes — drop to Qwen2.5-0.5B by editing cell 3 of notebook 01 (`BASE_MODEL = 'unsloth/Qwen2.5-0.5B-Instruct'`, `LORA_RANK = 16`). You won't beat Claude Haiku, but you'll see the reward-curve climb and produce defensible eval numbers within ~5h.

**Q: Can I use a non-Qwen model (e.g. Llama-3.2-3B)?**
A: Yes. Replace `BASE_MODEL` with any HF model ID Unsloth supports (see [Unsloth's supported models list](https://docs.unsloth.ai/get-started/all-our-models)). Qwen 2.5 is recommended because the SFT corpus uses ChatML formatting that Qwen handles natively.

**Q: Where does the trained adapter end up?**
A: `dakshdoesdev/sre-gym-qwen25-3b-grpo` on HF Hub by default (cell 9 of notebook 01). Change `REPO_ID` in cell 9 to push elsewhere.

**Q: My team has 3× HF Pro Spaces. Can we parallelize training?**
A: Three independent training runs with different seeds is the simplest parallelism. Real distributed training across machines requires the OpenClaw-RL pool-server (see §10) — point all 3 trainers at one shared pool server, get 3× rollout throughput. The GRPO group-relative advantages still work because each trainer batches its own K rollouts independently.

**Q: How do I add a domain (e.g. data-pipeline SRE) without restructuring?**
A: Add new templates to `EXTRA_TEMPLATES` in [`basic_templates_extra.py`](unified_incident_env/server/basic_templates_extra.py). The simulator core handles arbitrary 4-service topologies; extend `ServiceName` Literal in `models.py` if you need new service IDs. For Advanced-tier domain expansion, write new YAMLs in `sre_gym/advanced/scenarios/` — no Python changes required.

**Q: Can I run the env without OpenEnv?**
A: The Pydantic models and the simulator core (`UnifiedIncidentEnvironment`) are independent of the OpenEnv HTTP wiring — you can drive them directly. The HTTP routes (`/reset`, `/step`, `/state`) are just OpenEnv's contract layer. If you're integrating with a non-OpenEnv RL framework, import `unified_incident_env.UnifiedIncidentEnvironment` directly.

**Q: How does the Claude Code skill differ from `inference.py`?**
A: `inference.py` is a one-shot OpenAI-client script that runs a model against the env via the Hugging Face Inference Router. The skill is interactive: it lets you drive an episode via Claude Code's tool-use loop, with runbook persistence and incremental learning. Use `inference.py` for batch eval; use the skill for exploration / debugging / runbook-building.

**Q: Why is the per-step reward so small (-0.01 to +0.05)?**
A: Step rewards are deliberately small. The terminal `successful_resolution_bonus` (0.25) is what dominates the trajectory return. Per-step shaping is only for credit assignment — distinguishing "this query advanced potential" from "this query didn't" — not for being the primary signal.

**Q: How do I see the per-template eval breakdown?**
A: After running notebook 02, open `eval/results/comparison_per_template.png` for box-and-whisker plots, or load `comparison_summary.csv` into pandas:

```python
import pandas as pd
df = pd.read_csv('eval/results/comparison_raw.csv')
df.groupby(['template_id', 'policy'])['final_score'].agg(['mean', 'std', 'count'])
```

---

For deeper-than-execution-runbook material, see:

- [`README.md`](README.md) — full pitch + architecture overview
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — dimensional-escalation defence
- [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md), [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md), [`docs/MAX_TIER.md`](docs/MAX_TIER.md) — per-tier deep dives
- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — composable rubric design
- [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) — full scenario-authoring guide
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — postmortem corpus + benchmarking literature
