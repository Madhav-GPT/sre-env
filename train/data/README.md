# Teacher trajectories for SFT warm-start

**The canonical training input is `seed_combined.jsonl`** (21 episodes, 204 raw steps, **200 trainable** after empty-prompt filter). It merges three teachers with deliberately different characteristics:

| Source | Episodes | Resolved | Mean score | Role |
|---|---|---|---|---|
| Claude Opus 4.7 (hand-driven via pool server) | 6 | 6/6 | 0.769 | **Expert demos.** Author-optimal paths, full verification, observation-only (no runbook). Recorded 2026-04-24. |
| Llama-3.3-70B-Instruct via Fireworks | 4 | 3/4 | 0.725 | **Solid agent.** Usually picks the right rollback target, sometimes overshoots or misses a check. Recorded 2026-04-25. |
| Llama-3.3-70B-Versatile via Groq free tier | 11 | 5/11 | 0.421 | **Noisy realistic agent.** Often loops on query/hypothesis without committing to rollback — the exact failure mode GRPO needs to fix. Recorded 2026-04-25. |

**Why three teachers with different scores is deliberate**: Claude teaches format + optimal paths; Fireworks-Llama provides the middle band (what a trained 7B should plausibly match); Groq-Llama provides the "what not to do" lower band. GRPO needs samples across the reward distribution to estimate advantages — a corpus of only-expert-demos would make GRPO flat because every advantage would be ~0.

## Files

- `seed_combined.jsonl` — **canonical training corpus** (input for `sanity_run.ipynb`)
- `claude_seed.jsonl` — 6 Claude episodes only (provenance)
- `llama33_70b_smoke4.jsonl` — 4 Fireworks episodes
- `llama33_70b_groq_smoke3.jsonl` — 3 Groq smoke-test episodes
- `llama33_70b_groq_100.jsonl` — 8 Groq production-run episodes (stopped early at 8 when free-tier TPM capped further progress)
- `claude_<scenario>.jsonl` — raw per-episode event logs from the Claude run (auditable reset / step / evaluate events)

## Scenario coverage

All 6 scenario templates represented in `seed_combined.jsonl`:

| Template | Teacher episodes |
|---|---|
| worker_deploy_cascade | Claude ×1, +1 procgen; Groq ×5 variants |
| db_config_rollout | Claude ×1, +1 procgen; Fireworks ×1; Groq ×4 variants |
| gateway_auth_rollout | Claude ×1, +1 procgen; Fireworks ×1 |
| payment_webhook_misconfig | Fireworks ×1; Groq ×1 |
| schema_drift_missing_migration | Groq ×1 |
| cache_stale_state | Fireworks ×1 |

## Gotchas

- **Filter `len(prompt) < 50`** in the loader: 4 of the Claude rollback steps lost their prior observation to a chained-call logging bug. Reference implementation in `train/sanity_run.ipynb` cell 10.
- **Fireworks free-tier daily quota**: tight. After ~20 episodes at parallelism=3 the account hits a hard global 429 that persists until UTC midnight.
- **Groq free-tier TPM cap**: 6K tokens/min for Llama-3.3-70B-Versatile. Collection stalls around 8-10 episodes. Workarounds: (a) switch to `llama-3.1-8b-instant` which has higher TPM, (b) wait for TPM window reset, (c) upgrade Groq to paid tier.

## Reproduce / extend

```bash
# 1. Boot env (local or live HF Space)
python -m uvicorn unified_incident_env.server.app:create_compatible_app --factory --port 8000

# 2. Collect more Groq teacher episodes (run multiple times if TPM stalls)
export GROQ_API_KEY=...
python train/collect_trajectories.py \
  --env-url http://127.0.0.1:8000 \
  --scenarios all \
  --models "llama-3.3-70b-versatile" \
  --episodes-per-model 50 \
  --parallelism 2 \
  --driver groq \
  --output train/data/llama33_70b_groq_more.jsonl

# 3. Re-merge (deterministic)
cat train/data/claude_seed.jsonl \
    train/data/llama33_70b_*.jsonl > train/data/seed_combined.jsonl
```
