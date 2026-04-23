# sre-gym — 60-second pitch

> You can't train SRE agents on production. We built the gym.

## The story (00:00–01:00)

**[0:00–0:10 · Hook]** "Most SRE agent skills are prompts — a runbook and a good intention. We built the other half: a fault-injecting environment with deterministic grading, where every run is scored the same way twice."

**[0:10–0:25 · What it is]**
- OpenEnv-compliant. `openenv validate` passes.
- Three curriculum scenarios, easy → hard:
  - **easy** `worker_deploy_cascade` — bad worker deploy cascades to a DB crash.
  - **medium** `db_config_rollout` — DB config shrank the connection pool; a recent worker deploy is a decoy.
  - **hard** `gateway_auth_rollout` — bad auth-middleware rollout; two plausible suspects, one right answer.
- 11 bounded actions, honest state transitions (rolling back the wrong thing *fails*), deterministic grader across recovery / containment / verification / impact / efficiency.
- 21 tests passing. One public Space URL.

**[0:25–0:55 · Live demo]** `./demo/run_demo.sh`
- Env starts. Three scenarios visible in `/tasks`.
- Runbook dir cleared; demo starts cold.
- Each scenario solves end-to-end (score ≈ 0.99, 8–10 steps).
- A markdown runbook is written per scenario from the successful trace.
- Re-solve the easy scenario — this time the skill loads the runbook first. Same score, same path, zero wasted investigation.
- Point to `skill/verified-runbooks/` — "Every clean solve makes the next one deterministic. No GRPO required for v1."

**[0:55–1:00 · Close]** "Install the skill by symlinking `skill/` into `~/.claude/skills/sre-gym`. Open source, Apache 2. v2 is the OpenClaw-RL loop — distill this corpus of verified runbooks into a local 3B reviewer."

## The one technical claim you should be ready to defend

> "The env is honest."

- No hidden oracles. Rolling back the wrong service returns a negative reward and `failure_type="wrong_remediation_target"` — same observation contract as any other action.
- `declare_resolved` is rejected until the scenario's `resolution_check` passes, verified by actual service states in the world model, not a flag the grader peeks at.
- Rewards reward *effects*, not evidence-gathering — you can't farm the env by spamming `query_logs`.
- `restart_service` on the database before the root cause is removed returns a negative reward. Always. Because in the real world, it would crash again.

## Judge Q&A cheat sheet

**"How is this different from running a real staging env?"**
Deterministic scoring. Every agent gets graded against the same signatures, same decoys, same tick budget. You can't do that on real infra.

**"Why only three scenarios?"**
Three clears the hackathon DQ gate (`easy/medium/hard`). Each has a decoy + causal chain — building another one is a data-entry exercise, not a design one. Adding scenarios #4–#20 is the v2 data scaling lane.

**"Why runbooks instead of GRPO?"**
For this submission, GRPO means 48 hours of training convergence risk on top of an env we just shipped. Markdown runbooks demonstrate the same loop (verified signal → persisted artefact → next run improves) in an auditable form. The GRPO wiring slots on top of the same traces when we're ready.

**"What's the skill actually doing at runtime?"**
The skill lives in `skill/SKILL.md`. It directs Claude (or any agent) to read `verified-runbooks/{scenario}.md` before the first action, drive the env through `skill/tools/sre_gym_client.py`, and append a fresh runbook on any solve with `final_score > 0.85`.
