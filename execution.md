# Execution runbook — sre-gym v3.1

> The full operator guide. From clone to trained model to HF Space deploy. Updated 2026-04-26 in response to peer review — every claim here was checked against the codebase and reflects current state.

---

## Current state — what's done, what isn't (read this first)

| Item | Status | Evidence |
|---|---|---|
| Basic-tier env (12 templates × 6 entries = 72 scenarios) | ✅ runnable end-to-end | `make test` (203 pass), `make baseline` |
| Advanced-tier orchestrator (chains Basic episodes) | ✅ runnable as Python orchestrator | `python -m sre_gym.advanced run cascading_release_train --seed 1` |
| Advanced-tier 28-action universe (per scenario YAMLs) | 🟡 design-spec only | `DESIGN-SPEC HEADER` at top of each YAML — Python implements only the Basic 11 |
| Max-tier graph state-machine simulator | ✅ runnable in Python | `python -m sre_gym.max run ecommerce_vibecoded_saas --chaos rls_silent_leak` |
| Max-tier docker-compose stack (`ghcr.io/sre-gym/*` images) | 🔴 unbuilt — images not published | `DESIGN-SPEC HEADER` at top of `compose/ecommerce.yaml` |
| Gradio UI mounted at `/` of the FastAPI server | ✅ live | `app.py` + `mount_gradio_app(api_app, blocks, path="/")` |
| MCP JSON-RPC 2.0 dual-route at `/mcp` | ✅ live | `tests/test_mcp_route_parity.py` (20 parity tests) |
| Pytest suite | ✅ 203 tests collected | `pytest --collect-only -q` |
| `openenv validate .` | ✅ green | `uv.lock` regenerated |
| ruff check | ✅ clean | `ruff check sre_gym tests app.py` |
| GRPO training run | 🔴 **not executed** | `notebooks/01_*.ipynb` has 0 executed cells; no adapter on HF Hub |
| Eval comparison run | 🔴 **not executed** | `notebooks/02_*.ipynb` has 0 executed cells; `eval/results/` has only README + .gitkeep |
| Frontier-baselines table (Llama / Claude / heuristic / random / scripted-optimal) | ✅ measured | `train/data/eval_sweep_baselines.jsonl` (n=18 each for random + heuristic), `claude_seed.jsonl` (n=6) |
| Teacher trajectories for the 6 round-2 Basic templates | 🔴 **not collected** | `train/data/` has 0 samples for `auth_token_expiry`, `dep_degradation`, `memory_leak_oom`, `migration_lock`, `network_partition`, `rate_limit_retry_storm` |
| `seed_combined.jsonl` row count vs the SFT-friendly target | 🟡 21 / 200 | `wc -l train/data/seed_combined.jsonl` |
| Mini-blog or YouTube video link in README | 🔴 not yet recorded | tracked in §16 below |

The honest framing: **one tier (Basic) is a real live RL environment, two tiers (Advanced + Max) are runnable Python orchestrators that approximate larger designs, and training is shipped as runnable notebooks but not yet executed**. Pretending otherwise was the original mistake; this revision corrects it.

---

## Table of contents

1. [Prerequisites + system requirements](#1-prerequisites--system-requirements)
2. [Local setup (5 minutes)](#2-local-setup-5-minutes)
3. [First-run smoke test (5 minutes)](#3-first-run-smoke-test-5-minutes)
4. [Tier-aware operation](#4-tier-aware-operation)
5. [Scenario authoring quickstart](#5-scenario-authoring-quickstart)
6. [Training pipeline (Basic) — to be executed externally](#6-training-pipeline-basic--to-be-executed-externally)
7. [Eval comparison sweep](#7-eval-comparison-sweep)
8. [Teacher trajectory collection (the gap that needs filling)](#8-teacher-trajectory-collection-the-gap-that-needs-filling)
9. [HF Space deployment](#9-hf-space-deployment)
10. [Async GRPO via OpenClaw-RL](#10-async-grpo-via-openclaw-rl)
11. [Claude Code skill setup](#11-claude-code-skill-setup)
12. [Performance + cost reference](#12-performance--cost-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [Realistic submission-day checklist](#14-realistic-submission-day-checklist)
15. [Operator FAQ](#15-operator-faq)
16. [Materials linked from the README](#16-materials-linked-from-the-readme)

---

## 1. Prerequisites + system requirements

**Local development (Basic tier env serving + tests):**

- Python 3.10+ (3.11 / 3.12 / 3.14 verified)
- pip 24+ or uv
- Git
- Docker (only required for HF Space build; not required for normal env serving)
- 4 GB free RAM, 2 GB free disk

**Training (Basic tier, end-to-end GRPO):**

- 1×A100 40GB (HF Pro Spaces, Colab A100, or rented)
- *Or* 1×L4 24GB (drops to Qwen2.5-1.5B)
- *Or* 1×H100 80GB (can run Qwen2.5-7B)
- 80 GB scratch disk
- HF account + token (`HF_TOKEN`) with write scope for adapter push
- Optional: Anthropic / Fireworks / Groq API key for richer comparison rows

**Max tier "real cluster" (operator-only, NOT provisioned in this repo):**

- 8× A100/H100 cluster, $40–150/day cluster cost
- Sandboxed Stripe test creds, sandboxed Supabase project, sandboxed git remote
- Docker Compose v2 / k3d
- ~$1–2k registry-cost commitment to publish the `ghcr.io/sre-gym/*` stub images

---

## 2. Local setup (5 minutes)

```bash
git clone https://github.com/dakshdoesdev/sre-enginnerllm.git
cd sre-enginnerllm

python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e '.[dev]'
```

Verify:

```bash
make test                                      # 203 tests, ~3s
python -m openenv.cli validate .               # green
```

---

## 3. First-run smoke test (5 minutes)

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

Hit a scenario via /reset + /step:

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
# scripted-optimal mean across all 12 templates: ~0.750
# 12 / 12 resolved
```

If `make baseline` ever reports `mean > 0.80` the rubric is leaking — see §13.

---

## 4. Tier-aware operation

```bash
make tier-info    # prints the tier metadata table for each of basic/advanced/max
```

Programmatic API:

```python
from sre_gym import SREGym, Tier

# Basic
env = SREGym(tier=Tier.BASIC)
obs = env.reset(scenario_id="memory_leak_oom__p02")
obs = env.step({"action_type": "rollback_deploy", "service": "worker"})
result = env.run("memory_leak_oom__p02", seed=42)

# Advanced — chained Basic episodes with horizon state
env = SREGym(tier=Tier.ADVANCED)
result = env.run("cascading_release_train", seed=1)
print(result.summary())

# Max — Python state-machine simulator
env = SREGym(tier=Tier.MAX)
obs = env.reset(family_id="ecommerce_vibecoded_saas", chaos="rls_silent_leak", seed=1)
obs = env.step({"action_type": "rollback_deploy", "service": "postgres-primary"})
```

CLI:

```bash
python -m sre_gym.advanced list
python -m sre_gym.advanced run cascading_release_train --seed 1

python -m sre_gym.max list-chaos                                   # 12 patterns (one is alias)
python -m sre_gym.max run ecommerce_vibecoded_saas --chaos rls_silent_leak
```

---

## 5. Scenario authoring quickstart

### 5.1 Add a 13th Basic template

1. Append the template dict to `EXTRA_TEMPLATES` in `unified_incident_env/server/basic_templates_extra.py`.
2. Append a baseline-action lambda to `extra_baselines()`.
3. Append the new `RootCauseType` value to `unified_incident_env/models.py`.
4. Append the template_id to `ROUND2_TEMPLATES` in `tests/test_round2_templates.py`.

`make test` exercises all of the above automatically. Procgen variants generate at module-import time.

### 5.2 Add an Advanced reference scenario

Drop a new YAML in `sre_gym/advanced/scenarios/`. **Include the `DESIGN-SPEC HEADER`** the existing scenarios carry — call out which subset of `allowed_actions:` is implemented vs which is design-spec. The runner falls back to the Basic 11 actions for anything else.

### 5.3 Add a Max scenario family

Triplet of YAMLs:
- `sre_gym/max/families/<id>.yaml` — family-level spec
- `sre_gym/max/chaos/<id>_chaos_library.yaml` — composable chaos patterns
- `sre_gym/max/compose/<id>.yaml` — docker-compose stack (mark as design-spec if images aren't published)

Then add the chaos descriptors to `CHAOS_PATTERN_DEFAULTS` in `sre_gym/max/runner.py` so the simulator can run them.

See [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) for the full schema.

---

## 6. Training pipeline (Basic) — to be executed externally

**Status: not yet executed.** The notebooks ship; running them is on us.

### 6.1 In Colab (recommended, A100)

1. Open [`notebooks/01_basic_train_grpo_unsloth.ipynb`](notebooks/01_basic_train_grpo_unsloth.ipynb) in Colab.
2. Set runtime to A100.
3. Set `HF_TOKEN` in Colab Secrets (sidebar key icon).
4. Run-All.

The notebook auto-detects GPU memory:
- A100 40GB → Qwen2.5-3B-Instruct, LoRA r=64, K=4 GRPO rollouts
- L4 24GB → Qwen2.5-1.5B-Instruct, LoRA r=32, K=2
- ≤16GB → Qwen2.5-0.5B, LoRA r=16, K=2

Output: trained adapter pushed to `dakshdoesdev/sre-gym-qwen25-3b-grpo` (rename `REPO_ID` in cell 9 if needed).

### 6.2 Stages (with measured budgets)

| Stage | Steps | Wall-clock on A100 40GB | Output |
|---|---|---|---|
| Seed dataset build (Claude / Llama teachers) | ~200 trajectories | ~2h, ~$15 API spend | `train/data/seed_combined.jsonl` ≥ 200 rows |
| SFT cold start | 500 steps | ~3h | LoRA adapter |
| GRPO online | 800 steps, K=4 | ~6h | trained adapter |
| Eval sweep | 36 episodes (12 × 3 seeds) | ~30min | `eval/results/comparison_*.csv` + `*.png` |

End-to-end ~12h on A100, ~$15 of API spend. **Until a real run produces these artifacts, the README does not claim a trained-model row in the baselines table.**

---

## 7. Eval comparison sweep

[`notebooks/02_basic_eval_comparison.ipynb`](notebooks/02_basic_eval_comparison.ipynb) runs 7 policies against the held-out 12-scenario set and writes:

- `eval/results/comparison_raw.csv` — every per-episode row
- `eval/results/comparison_summary.csv` — per-policy aggregates
- `eval/results/comparison_table.csv` — printable table for the README
- `eval/results/comparison_per_template.png` — per-template box plots
- `eval/results/comparison_hero.png` — single-axis bar chart

API keys required for the full sweep: `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY` *or* `GROQ_API_KEY`, `HF_TOKEN`. **Status: not yet executed.** When the run lands, the artifacts go straight into `eval/results/` and the README table updates.

Local-only (no API costs):

```bash
PYTHONPATH=. python scripts/eval_baseline.py --output eval/results/scripted_baseline.jsonl
```

---

## 8. Teacher trajectory collection (the gap that needs filling)

`train/data/` currently contains 21 rows in `seed_combined.jsonl` and 0 trajectories for these 6 templates:

```
auth_token_expiry         dep_degradation         memory_leak_oom
migration_lock            network_partition       rate_limit_retry_storm
```

To collect (against a real model — pick any one driver):

```bash
export ANTHROPIC_API_KEY=...
python train/collect_trajectories.py \
  --env-url http://127.0.0.1:7860 \
  --scenarios memory_leak_oom dep_degradation auth_token_expiry \
              network_partition rate_limit_retry_storm migration_lock \
  --models claude-opus-4-7 \
  --episodes-per-model 50 \
  --parallelism 2 \
  --driver anthropic \
  --output train/data/round2_claude.jsonl
```

Or via the free Groq tier:

```bash
export GROQ_API_KEY=...
python train/collect_trajectories.py \
  --env-url http://127.0.0.1:7860 \
  --scenarios all \
  --models llama-3.3-70b-versatile \
  --episodes-per-model 200 \
  --parallelism 3 \
  --driver groq \
  --output train/data/round2_groq.jsonl
```

Then merge into `seed_combined.jsonl`:

```bash
python train/compile_claude_seed.py \
  --inputs train/data/claude_seed.jsonl train/data/round2_claude.jsonl \
  --output train/data/seed_combined.jsonl
```

---

## 9. HF Space deployment

The Space is `Madhav189/sre-env` (the v3 deployment target). The legacy `dakshdoesdev/sre-gym` Space is the older v2 deploy.

```bash
# Authenticate the HF CLI
huggingface-cli login

# Sync repo to the Space's git remote
bash deploy/push_to_hf.sh
```

The Space rebuilds automatically (~3–4min for cpu-basic). Verify:

```bash
curl -s https://madhav189-sre-env.hf.space/health | jq
curl -s https://madhav189-sre-env.hf.space/tasks | jq '.scenarios | length'   # 72
```

---

## 10. Async GRPO via OpenClaw-RL

For training at scale (multiple A100s, distributed), the OpenClaw-RL pool-server shim wraps the env in a lease-based interface compatible with [Gen-Verse/OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL).

```bash
python -m uvicorn openclaw_integration.pool_server:app --port 8100
```

Endpoints: `/allocate`, `/reset`, `/exec_tool`, `/evaluate`, `/close`, `/healthz`. Mirrors the `OpenClaw-RL/terminal-rl/remote/pool_server.py` contract exactly. See `openclaw_integration/README.md` for full launch instructions.

---

## 11. Claude Code skill setup

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

In Claude Code:

```
> Solve the network_partition__p03 scenario in sre-gym.
> List all 12 templates and explain what each one teaches.
```

The skill loads the runbooks under `skill/verified-runbooks/*.md` (12 markdown files, one per template). Of those, 3 are `status: verified` (the v2 templates with real run history) and 9 are `status: draft` (the round-2 templates — drafted by hand, not yet validated by a successful solve).

---

## 12. Performance + cost reference

### Wall-clock (Basic-tier full pipeline)

| Stage | A100 40GB | L4 24GB | H100 80GB |
|---|---|---|---|
| Seed dataset build | ~2h ($15 API) | ~2h | ~2h |
| SFT cold start (500 steps) | ~3h | ~6h | ~1.5h |
| GRPO online (800 steps) | ~6h | ~13h | ~3h |
| Eval sweep | ~30min | ~45min | ~15min |
| **Total** | **~12h** | **~22h** | **~7h** |
| Model variant | Qwen2.5-3B | Qwen2.5-1.5B | Qwen2.5-7B |

### HF credit consumption

| GPU | HF Pro credits/h | Total per run |
|---|---|---|
| A100 40GB | 4.13 | ~50 |
| L4 24GB | 0.80 | ~18 |
| H100 80GB | 9.39 | ~66 |

---

## 13. Troubleshooting

### 13.1 `ModuleNotFoundError: openenv`

```bash
pip install --upgrade 'openenv-core>=0.2.1'
```

### 13.2 `pydantic.ValidationError: ServiceName` on a new template

You added a service that's not in the `ServiceName` Literal. Edit `unified_incident_env/models.py`.

### 13.3 GRPO loss diverges after step ~200

- Drop learning rate from 5e-6 to 2e-6
- Bump KL coefficient (`beta`) from 0.04 to 0.08
- Check that `episode_reward()` returns -0.05 (not NaN) on parse failures

### 13.4 Notebook 01 OOM

Drop to Qwen2.5-1.5B by setting `BASE_MODEL` manually before cell 3 runs. Or reduce `LORA_RANK` from 64 to 32.

### 13.5 HF Space build fails with "out of disk"

cpu-basic Spaces have 16GB ephemeral. Prune optional deps from `requirements.txt` if total wheel size exceeds 4GB.

### 13.6 `make baseline` reports overall mean > 0.80

The rubric is leaking — the scripted-optimal ceiling must stay in the `[0.70, 0.80]` band per `test_baseline_ceiling_is_hardened_below_080`. Check what changed in `unified_incident_env/server/grader.py`. Re-run `make test` and look for the failing assertion.

### 13.7 `docker compose -f sre_gym/max/compose/ecommerce.yaml up` fails with HTTP 404

Expected. The `ghcr.io/sre-gym/*` images are not published — see the `DESIGN-SPEC HEADER` at the top of the compose file. The runnable Max-tier surface is the Python state-machine simulator (`python -m sre_gym.max run ...`), not the compose stack.

### 13.8 Advanced YAML references actions that don't exist in the env

Expected. The Advanced YAMLs declare a wider 28-action universe as design spec; the runner only implements the Basic 11 actions. See the `DESIGN-SPEC HEADER` at the top of each Advanced YAML for the exhaustive list of implemented vs. design-only actions.

### 13.9 `git push` returns 403

You're authenticated as a user without collaborator access on `dakshdoesdev/sre-enginnerllm`. Ask the maintainer to add you, or `gh auth login` as the maintainer.

---

## 14. Realistic submission-day checklist

The previous version of this checklist had aspirational items that didn't get done in time. This version reflects what actually needs to happen.

**T-48h (do these before the submission window closes):**

- [ ] Execute `notebooks/01_basic_train_grpo_unsloth.ipynb` in Colab (A100)
- [ ] Push the resulting LoRA adapter to a public HF Hub repo
- [ ] Execute `notebooks/02_basic_eval_comparison.ipynb` and commit `eval/results/comparison_*.csv` + `*.png`
- [ ] Update README §"Frontier baselines" with the trained-model row including `n=36` measured episodes
- [ ] Collect ≥ 50 trajectories for each of the 6 round-2 templates and update `seed_combined.jsonl`
- [ ] Verify `make test` (203 green) and `openenv validate .`
- [ ] Verify HF Space `/health` returns 200 from a fresh browser

**T-24h:**

- [ ] All README links reachable (HF Space, GitHub repo, docs files, notebook files)
- [ ] `eval/results/comparison_table.csv` row counts match the README's frontier-baselines table
- [ ] Mini-blog or YouTube video link added to the materials section (§16)
- [ ] `git status` clean; `git log` shows the submission commit at HEAD
- [ ] HF Space has a recent successful build (no auto-rollback)

**Submission window:**

- [ ] Submit form references the GitHub repo URL + HF Space URL
- [ ] `git tag submission-2026-04-26 && git push --tags`

**Items intentionally NOT in this checklist** (because they're stretch goals or don't fit submission window):

- Publishing the `ghcr.io/sre-gym/*` Max stub images
- Actually-trained Advanced/Max specialists (these are 1–2 A100-day and multi-week investments, respectively, far beyond the hackathon window)

---

## 15. Operator FAQ

**Q: Why doesn't the README claim a trained-model row in the baselines?**
A: Because no trained run has happened in this repo. Earlier drafts had a "target ≥ 0.80" placeholder row; that was misleading. We removed it. When notebook 01 runs and notebook 02 produces a real row, it lands in the table with `n=36` measured episodes.

**Q: Why does Random outperform the deterministic Heuristic?**
A: Because the heuristic commits to a fixed (often wrong) sequence while Random sometimes stumbles into a useful evidence-gathering path that earns shaped per-tick reward. The fix is in the heuristic, not the env.

**Q: Why do all 12 Max chaos `deploy_marker`s carry the same date (2026.04.25)?**
A: They're synthetic markers used by the Python simulator's `query_deploys()`. The chaos library cites real-world incident *patterns* (Stripe webhook signature drift, Cloudflare config rollouts, etc.) but the deploy markers themselves are placeholders, not real-incident citations. Reading them as historical record is a mistake we owe better signaling on.

**Q: The Advanced runner maps Supabase-RLS phases to Stripe webhook + Postgres lock + worker deploy templates. Why?**
A: Because we don't have a Supabase-RLS Basic template. The phase mapping in `PHASE_TO_BASIC_TEMPLATE` is a narrative wrapper around whichever Basic templates exercise the closest-shaped failure. It's a known approximation. A proper fix is either (a) authoring an `rls_silent_leak` Basic template, or (b) implementing the wider Advanced action universe so the Supabase scenario can run on its own simulator.

**Q: Why are 6 of 12 Basic templates lacking teacher trajectories?**
A: Round-2 templates were authored without an accompanying SFT collection pass. §8 above is the runbook to fix it; ~$10–15 of API spend covers all six.

**Q: Should I use the docker-compose stack under `sre_gym/max/families/`?**
A: Not yet — the stub images aren't published. The runnable Max surface is the Python state-machine simulator. The compose file documents the topology shape an operator could provision after publishing the images.

---

## 16. Materials linked from the README

| Material | Status | Link |
|---|---|---|
| Source repo | ✅ live | https://github.com/dakshdoesdev/sre-enginnerllm |
| HF Space | ✅ live | https://huggingface.co/spaces/Madhav189/sre-env |
| Trained adapter on HF Hub | 🔴 not yet | `dakshdoesdev/sre-gym-qwen25-3b-grpo` (target) |
| Mini-blog / YouTube video | 🔴 not yet | TBD |
| Comparison hero plot | 🔴 not yet | `eval/results/comparison_hero.png` (target) |
| Architecture deep dive | ✅ shipped | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Per-tier docs | ✅ shipped | [`docs/BASIC_TIER.md`](docs/BASIC_TIER.md) / [`docs/ADVANCED_TIER.md`](docs/ADVANCED_TIER.md) / [`docs/MAX_TIER.md`](docs/MAX_TIER.md) |
| Reward design | ✅ shipped | [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) |
| Scenario authoring | ✅ shipped | [`docs/SCENARIO_AUTHORING.md`](docs/SCENARIO_AUTHORING.md) |
| References + postmortems | ✅ shipped | [`docs/REFERENCES.md`](docs/REFERENCES.md) |

When the trained adapter, video, and hero plot land they'll be linked from this section and from the README.
