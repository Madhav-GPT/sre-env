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

# sre-gym — Fault-injecting SRE training env for OpenEnv

Most SRE agent skills are runbooks and good intentions. **sre-gym** is the other half: a fault-injecting environment with deterministic grading where an agent diagnoses a real production-style incident, chooses a safe remediation, verifies recovery, and declares resolved. Every run is scored the same way twice.

- Spec-compliant OpenEnv environment (typed Pydantic action / observation / state, `reset` / `step` / `state`, `openenv validate` green).
- 3 curriculum scenarios — easy, medium, hard — with decoy services and causal dependencies.
- 11 bounded actions. Honest state transitions. No hidden oracles.
- 21 tests passing.
- Ships a Claude Code skill + verified-runbook loop — successful solves write markdown runbooks that the next run reads back.

## 30-second demo

```bash
./demo/run_demo.sh
```

Starts the env, solves each scenario cold, writes a runbook for each, re-solves to prove the loop. Full transcript takes ~10 seconds.

## Curriculum

| Difficulty | Scenario | Story | Decoy | Correct path |
|---|---|---|---|---|
| easy | `worker_deploy_cascade` | Bad worker deploy → DB crash-loop → login 502s | — | rollback worker → restart db → verify → resolve |
| medium | `db_config_rollout` | DB config push shrank connection pool from 80→12 | recent worker deploy | rollback **db** → restart db → verify → resolve |
| hard | `gateway_auth_rollout` | Gateway auth-middleware rollout rejects valid logins | recent worker deploy | rollback **gateway** → verify → resolve (no restart) |

Rolling back the wrong service returns a negative reward and `failure_type="wrong_remediation_target"`. Restarting before the cause is removed re-inherits the bad state. `declare_resolved` is rejected until the scenario's resolution check passes against the actual world model.

## Install

```bash
# 1. Create a venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# 2. Start the env
uvicorn server.app:app --host 127.0.0.1 --port 8000

# 3. Run the baseline inference against it
export HF_TOKEN="…"; export ENV_BASE_URL=http://127.0.0.1:8000
python inference.py
```

## Install the Claude Code skill

```bash
ln -s "$PWD/skill" "$HOME/.claude/skills/sre-gym"
```

Then, in Claude Code, ask: *"Solve the db_config_rollout scenario in sre-gym."* The skill will drive the env via `skill/tools/sre_gym_client.py`, load any existing runbook from `skill/verified-runbooks/`, and append a fresh runbook on any clean solve (score > 0.85).

## Architecture

```
┌────────────────────┐      HTTP / WS       ┌──────────────────────┐
│  Claude Code       │ ──────────────────▶ │  OpenEnv server       │
│  (with sre-gym     │ ◀────────────────── │  (FastAPI, uvicorn)   │
│   skill loaded)    │    obs, reward      │  unified_incident_env │
└────────────────────┘                     └──────────────────────┘
        │                                            ▲
        ▼ on clean solve (score > 0.85)              │
┌────────────────────┐                               │
│ verified-runbooks/ │ ────── loaded at skill load ──┘
│   *.md             │
└────────────────────┘
```

## Scoring

Deterministic, 5 dimensions, sums to a public score in `[0.01, 0.99]`:

- **Recovery** (0–0.4): critical-path services healthy
- **Containment** (0–0.3): root cause removed or offending service isolated
- **Verification** (0–0.35): `database_recovery` + `end_to_end` checks passed
- **Impact** (0–0.15): user-impact reduced
- **Efficiency** (0–0.10): budget preserved, no wasteful repeats

Target **> 0.85** for "clean solve." That's also the runbook-record threshold.

## Repo layout

```
unified_incident_env/    # env core: models, environment, grader, challenge, tests
server/                  # OpenEnv entrypoint wrapper
skill/                   # Claude Code skill: SKILL.md, tools/, verified-runbooks/
demo/                    # run_demo.sh + pitch.md
inference.py             # OpenAI-client baseline for OpenEnv hackathon submission
openenv.yaml             # OpenEnv manifest
Dockerfile               # HF Space deployment
```

## Verify

```bash
pytest unified_incident_env/tests -q          # 21 tests
python -m openenv.cli validate .              # OpenEnv manifest check
docker build -t sre-engineer-llm:v2 .         # HF Space image
```

## Roadmap — v2

Distill the accumulated `verified-runbooks/` corpus into a local 3B reviewer via [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)'s async GRPO-on-next-state loop. Same reward contract (`run_check` passes / `failure_type` absent), same grader, but a compact policy that runs without a frontier API.

## License

Apache 2.0
