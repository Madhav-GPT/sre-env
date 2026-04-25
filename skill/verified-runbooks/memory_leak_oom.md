---
template_id: memory_leak_oom
status: draft
last_verified: null
---

# Worker Memory Leak / OOM Restart Loop

## Symptoms

Worker pod restart count is climbing fast (e.g. 14 restarts in 20 minutes vs baseline 0). Worker shows `OOMKilled` exit codes. Database connection-establish rate spikes every ~90s in lockstep with the restart cadence. Gateway 5xx rate has a sawtooth pattern that aligns 1:1 with worker restart timestamps.

The trap: database CPU is loud (78%+) and looks like a database fault. It isn't — DB CPU is reactive, driven by the connection-establish bursts every time worker comes back from OOM. Worker CPU is *low* (12%) because the process spends most of its life dead, not working. The signal you must lean on is **restart count > error count**.

## Decision tree (preconditions → action)

1. If worker restart count ≥ 5 in last 30 min → strong signal for OOM loop.
2. If DB CPU spike pattern aligns to worker restart cadence → DB load is reactive.
3. If a recent worker deploy mentions "cache", "buffer", "prefetch", or "in-memory" → likely leak vector.
4. If confirmed → roll back the worker deploy, then restart database to drain connection backlog.

## Action sequence (what to call, in order)

1. `query_logs(worker)` — see OOMKilled exits and the deploy version that introduced the leak.
2. `query_metrics(worker, cpu)` — confirm worker CPU is *low*, ruling out compute as the cause.
3. `query_deploys(worker)` — confirm the leaking deploy and its timestamp.
4. `query_metrics(database, cpu)` — confirm DB load follows the restart cadence (reactive, not primary).
5. `submit_hypothesis(memory_leak_runaway, [worker, database, api-gateway], 0.85, rollback_deploy)`.
6. `rollback_deploy(worker)` — revert the leaking deploy.
7. `restart_service(database)` — drain the connection-establish backlog cleanly.
8. `run_check(database_recovery)` — confirm DB connection-establish rate returns to baseline.
9. `run_check(end_to_end)` — confirm gateway → worker → DB path completes.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Worker restart count is stable; no new OOMKilled exits.
- Worker `memory_pct` lands in the 30-45% band (not climbing).
- Database `cpu_pct` settles to baseline (~30-40%).
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back the *database* (instead of the worker) does nothing — there's no DB deploy to revert. Restarting the database *before* rolling back the worker means worker will OOM again 90s later and hammer the freshly restarted DB. Isolating worker reduces blast radius but blocks user traffic.
