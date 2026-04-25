# sre-gym — 60-second pitch (v3.1, honest)

> You can't train SRE agents on production. We built the gym.

This pitch was rewritten 2026-04-26 to match what's actually in the repo. An earlier v1 draft of this file claimed "21 tests passing" and "three curriculum scenarios"; both numbers are stale.

## The story (00:00–01:00)

**[0:00–0:10 · Hook]** "Most SRE-agent demos are prompts dressed up as products. We built the other half — a fault-injecting environment with deterministic grading, scored the same way twice across every run."

**[0:10–0:25 · What it is]**
- OpenEnv-compliant. `openenv validate` green; **203 tests** collected via `pytest --collect-only -q`.
- **One runnable RL environment** (Basic tier): 12 base templates, each with 5 procgen variants → **72 deterministic scenarios** over a 4-service topology.
- **Two design-spec tiers** with runnable Python orchestrators:
  - **Advanced** chains Basic episodes together with persistent horizon state (unresolved alerts, pending deploys, tech-debt counter, horizon-decay reward) — 3 reference scenarios.
  - **Max** is a Python state-machine simulator over a 22-node service graph — 12 chaos patterns. (The docker-compose stack alongside it is design-spec only; the stub images aren't published.)
- **11 typed actions** validated by Pydantic. 7-dimension grader. Hardened scripted-optimal ceiling enforced at `[0.70, 0.80]` by a CI test, leaving 0.20 of headroom for a trained agent.

**[0:25–0:55 · Live demo]** Two paths a viewer can pick:

1. **Live env over HTTP (Basic tier)** — `uvicorn app:app --port 7860`. Hit `/reset` then `/step`; the Gradio terminal at `/` walks any of the 12 templates with full observation streaming.
2. **CLI smoke (all three tiers)** — `make baseline` (Basic, all 12 templates) · `python -m sre_gym.advanced run cascading_release_train --seed 1` (Advanced, 2-phase chained-incident) · `python -m sre_gym.max run ecommerce_vibecoded_saas --chaos rls_silent_leak` (Max, security-classified chaos pattern over the in-memory graph).

**[0:55–1:00 · Honesty closer]** The notebooks for GRPO training (`notebooks/01_*`) and the comparison sweep (`notebooks/02_*`) ship runnable but **have not yet been executed in this repo**. We're running them on Colab post-submission-window and committing the trained adapter + plots to `eval/results/` when they land. The README does not claim a trained-model row in the baselines table until that happens. Frontier-LLM rows (Llama-3.3-70B / Claude Opus / scripted / random / heuristic) are real and measured; sample sizes are uneven (smoke-test collection) and we say so out loud.

## What's actually in the repo right now

| Component | Status |
|---|---|
| Basic-tier env with 72 scenarios + 11 actions + 7-dim grader | ✅ runnable |
| Gradio terminal UI mounted at `/` | ✅ runnable |
| MCP JSON-RPC 2.0 dual-route at `/mcp` | ✅ runnable + parity-tested |
| Advanced-tier orchestrator (chained Basic episodes) | ✅ runnable |
| Max-tier graph state-machine simulator | ✅ runnable |
| Advanced-tier wider 28-action universe | 🟡 design-spec only (each YAML carries a `DESIGN-SPEC HEADER`) |
| Max-tier docker-compose stack | 🟡 design-spec only (`ghcr.io/sre-gym/*` images not published) |
| GRPO training run + adapter on HF Hub | 🔴 pending external execution |
| Eval comparison sweep + plots | 🔴 pending external execution |
| Teacher trajectories for the 6 round-2 templates | 🔴 not yet collected (~$15 API spend covers it) |
| Mini-blog / video | 🔴 not yet recorded |

## Judge Q&A — anticipated questions, honest answers

**"Where are the training plots?"** Pending. Notebooks ship runnable; we're executing them externally on Colab and committing artifacts to `eval/results/` when they land. Until then there's no trained-model row in the baselines table.

**"Why does Random outperform your Heuristic in the baselines table?"** Because the heuristic commits to a fixed wrong sequence on most templates while Random sometimes stumbles into a useful evidence-gathering path that earns shaped per-tick reward. The fix is in the heuristic; we left it documented rather than buried.

**"Why do all 12 Max chaos patterns name the failing service in the incident_summary string?"** Because the simulator is a fault-injection harness, not a hidden-information puzzle. The README documents this openly. A real-cluster Max tier would use raw Loki/Tempo signals; the Python sim doesn't claim to.

**"The Advanced YAMLs reference 42 actions that don't exist in the Python codebase. What's going on?"** Each YAML carries a prominent `DESIGN-SPEC HEADER` listing the implemented 11 actions explicitly. The wider universe is documented as the design space a downstream operator (1–2 A100-days of focused engineering) would target. The runner falls back to the Basic 11 actions today.

**"Why is the Supabase RLS scenario backed by `payment_webhook_misconfig` + `migration_lock` + `worker_deploy_cascade`?"** Because we don't have a Supabase-RLS Basic template and the Advanced runner approximates higher-tier scenarios via the closest-shaped Basic templates. It's documented as an approximation, not as fidelity.

**"Why is the env honest then?"** Same reasons it always was:
- No hidden oracles. Rolling back the wrong service returns negative reward + `failure_type="wrong_remediation_target"` — same observation contract as any other action.
- `declare_resolved` rejected until the scenario's `resolution_check` passes, verified against actual service states in the world model.
- Rewards reward *effects*, not evidence-gathering — you can't farm the env by spamming `query_logs`.
- `restart_service` on the database before the root cause is removed returns negative reward. Always. Because in the real world it would crash again.

## The single defended sentence

**Each tier escalates a different dimension: Basic escalates compute, Advanced escalates horizon, Max escalates realism.** The framing is the load-bearing claim. Basic exists as a runnable RL environment; Advanced and Max are runnable orchestrators that honestly approximate larger designs that the YAMLs document in full.
