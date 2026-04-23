---
name: sre-gym
description: SRE incident-response training environment with fault injection and deterministic grading. Use when the user wants to practice SRE skills, solve an injected production incident, or run one of three scenarios (worker_deploy_cascade / db_config_rollout / gateway_auth_rollout) against the sre-gym HTTP server. Invokes scripts in skill/tools/ to query the env and records verified runbooks after clean solves.
---

# SRE Gym — Incident Response Skill

You are an SRE agent connected to a running sre-gym environment (HTTP, default `http://127.0.0.1:8000`). The env simulates production incidents with decoy services, deterministic grading, and explicit resolution checks. Your job is to diagnose from evidence, pick the correct remediation, verify recovery, then declare resolved.

## When to use this skill

- The user names a scenario (`worker_deploy_cascade`, `db_config_rollout`, `gateway_auth_rollout`) or says "solve an incident / run SRE scenario"
- The user asks you to practice, benchmark, or demo incident response
- The user points you at an sre-gym URL

## Core rules (never break these)

1. **Never guess at remediation.** Query evidence (`query_logs`, `query_deploys`, `query_metrics`) before `rollback_deploy` / `restart_service`.
2. **Root cause before restart.** Restarting a service before rolling back the triggering change re-inherits the bad state.
3. **Never call `declare_resolved` before the scenario's resolution check passes.** Each scenario specifies which check is required; read it from `observation.checks` and from any loaded runbook.
4. **Watch for decoys.** Each scenario has a plausible-looking wrong answer. Example: `db_config_rollout` has a recent worker deploy that is *not* the cause. Read logs before committing to a target.
5. **Repeating the same no-progress action wastes ticks.** The env emits `loop_warning` when you do this — treat it as a hard signal to try a different evidence source.

## Workflow

### 1. Load prior knowledge

Before your first action, check `skill/verified-runbooks/{scenario_id}.md`. If it exists, read it — it's a log of previously-successful solves for this exact scenario, written by earlier runs of this skill. Use the winning path and the decoy list.

### 2. Drive the env

Use `skill/tools/sre_gym_client.py` to call the env:

```bash
python skill/tools/sre_gym_client.py list           # show available scenarios
python skill/tools/sre_gym_client.py reset <id>     # start an episode
python skill/tools/sre_gym_client.py step '<json>'  # take one action
python skill/tools/sre_gym_client.py status         # current obs + grader
```

Action JSON matches the env's `UnifiedIncidentAction` model. Examples:
```json
{"action_type": "query_logs", "service": "database"}
{"action_type": "query_deploys", "service": "worker"}
{"action_type": "rollback_deploy", "service": "database"}
{"action_type": "run_check", "check_name": "end_to_end"}
{"action_type": "declare_resolved"}
```

### 3. Investigation loop (per tick)

1. Read `observation.prompt_text` — services, alerts, last result, failure_type, why_failed.
2. If `observation.failure_type` is set, your previous action was rejected — **do not repeat it**, read `why_failed` and pick a different evidence source or remediation.
3. Form a hypothesis with `submit_hypothesis` once you have enough evidence (usually 2–4 queries). Calibrate `confidence`: ≥0.7 only if you're sure.
4. Remediate (`rollback_deploy` → `restart_service` if scenario requires → `run_check`).
5. `declare_resolved` only after the required check passes.

### 4. Record the runbook

If the episode finishes with `incident_resolved=true` and `final_score > 0.85`, run:

```bash
python skill/tools/sre_gym_client.py record-runbook <scenario_id>
```

This appends a new entry to `skill/verified-runbooks/{scenario_id}.md`. Future runs of this skill (yours or another Claude's) load it automatically.

## Action reference (11 actions)

| Action | Required fields | Purpose |
|---|---|---|
| `query_logs` | `service` | Read service-level error logs |
| `query_metrics` | `service`, `metric` (cpu/error_rate/latency) | Read quantitative signals |
| `query_dependencies` | `service` | Map upstream/downstream |
| `query_deploys` | `service` | Recent deploy history |
| `rollback_deploy` | `service` | Revert last deploy — SCENARIO-SPECIFIC TARGET |
| `restart_service` | `service` | Reboot a service (usually after rollback) |
| `run_check` | `check_name` (`database_recovery` / `end_to_end`) | Objective recovery check |
| `isolate_service` | `service` | Containment only, does not resolve |
| `escalate` | — | Record escalation note |
| `submit_hypothesis` | `hypothesis` object | Commit RCA with confidence calibration |
| `declare_resolved` | — | Finalize; rejected if required check has not passed |

## Scoring rubric (deterministic from the env)

- **Recovery (0–0.4):** services healthy on the critical path
- **Containment (0–0.3):** root cause removed OR offending service isolated
- **Verification (0–0.35):** both checks passed
- **Impact (0–0.15):** user_impact reduced
- **Efficiency (0–0.10):** budget preserved, no wasteful repeats

Clean solve target: **> 0.85**. That's the runbook-record threshold.

## Decoy knowledge (read before hypothesizing)

- `worker_deploy_cascade`: the only true cause; no decoys.
- `db_config_rollout`: the recent worker deploy is a **decoy**. Rolling back worker yields `wrong_remediation_target`.
- `gateway_auth_rollout`: the recent worker deploy (`worker@...-hotfix` — log-format tweak) is a **decoy**. The gateway auth rollout is the cause.

If you take a wrong remediation, the env returns `failure_type="wrong_remediation_target"` and a negative reward — **do not retry the same wrong target**, re-read the logs.
