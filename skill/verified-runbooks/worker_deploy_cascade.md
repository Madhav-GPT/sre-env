---
template_id: worker_deploy_cascade
status: verified
last_verified: 2026-04-23T22:01:29Z
verified_score: 0.99
verified_steps: 10
---

# Worker Deploy Cascade

## Symptoms

A bad worker deploy causes sustained database overload and login 502s at the gateway. Worker queue depth and retry volume spiked at the rollout boundary. Database CPU is pinned at 99% and the database process is crashing under repeated overload. Gateway latency p95 climbs to 600ms+ as upstream calls to worker time out.

The trap: nothing — this is the entry-level template, where the loudest service (worker) really is the cause. Use it to anchor your other diagnoses; if a *more complex* scenario looks just like this one, double-check before acting.

## Decision tree (preconditions → action)

1. If gateway 502s correlate with worker errors AND a recent worker deploy is timestamped just before the incident → strong signal for worker-deploy cascade.
2. If database CPU is pinned AND there's no recent DB deploy → DB is reactive to worker, not faulty.
3. If confirmed → roll back worker, then restart database to drain the connection backlog cleanly.

## Action sequence (what to call, in order)

1. `query_deploys(worker)` — confirm a recent rollout aligns with the incident start.
2. `query_logs(worker)` — see the error pattern + deploy version that caused it.
3. `query_metrics(database, cpu)` — confirm DB is overloaded as a downstream effect.
4. `query_dependencies(api-gateway)` — verify gateway depends on worker → DB.
5. `submit_hypothesis(bad_worker_deploy, [worker, database, api-gateway], 0.82, rollback_deploy)`.
6. `rollback_deploy(worker)` — remove the triggering change.
7. `restart_service(database)` — bring DB back cleanly after the root cause is gone.
8. `run_check(database_recovery)` — verify DB is no longer crashing.
9. `run_check(end_to_end)` — verify gateway traffic succeeds end-to-end.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Worker `error_rate_pct` <= 2%; gateway 5xx rate returns to baseline.
- Database `status == "healthy"`; CPU drops to baseline ~30-40%.
- Both checks pass; `incident_resolved == True`; `final_score` lands in [0.74, 0.99].

## Rollback / safety notes

Restarting the database *before* rolling back worker means worker re-saturates DB on its next request burst. Isolating worker stops the bleeding but leaves the bad deploy live — if a teammate triggers a worker scale-up later, the storm restarts.

---

## Run history

### Run 2026-04-23T22:01:29Z — Score 0.99 (10 steps)

Winning path:
1. `query_deploys(worker)` — Check whether any recent deploy aligns with the incident start
2. `query_logs(worker)` — Inspect worker logs because deploy timing and queue pressure suggest worker-originated harm
3. `query_metrics(database, cpu)` — Confirm that the database is overloaded as a downstream effect
4. `query_dependencies(api-gateway)` — Verify the gateway depends on the worker and database path
5. `submit_hypothesis(bad_worker_deploy)` — Commit a calibrated hypothesis before taking an invasive mitigation step
6. `rollback_deploy(worker)` — Remove the triggering change before restarting downstream services
7. `restart_service(database)` — Bring the database back cleanly after the root cause is removed
8. `run_check(database_recovery)` — Verify the database is no longer crashing
9. `run_check(end_to_end)` — Verify gateway traffic succeeds end-to-end
10. `declare_resolved` — Declare resolved only after objective checks pass

Checks passed: database_recovery, end_to_end.
