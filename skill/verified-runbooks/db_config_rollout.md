---
template_id: db_config_rollout
status: verified
last_verified: 2026-04-23T22:01:33Z
verified_score: 0.99
verified_steps: 10
---

# Database Config Rollout Regression

## Symptoms

Database connection acquire timeouts at 48% and climbing. Write-path requests return sustained 5xx. Worker write latency is elevated; retries are climbing. Database CPU is *moderate* (~62%) — this is *not* a compute-overload pattern. The signal is `connection acquire timeout` errors, not `out of memory` or `query plan`. A separate worker deploy landed around the same time and looks suspicious but is not the cause.

The trap: the worker deploy is a decoy. Worker errors are reactive to database pool exhaustion, not local faults. The actual root cause is a recent DB config push that shrunk `max_connections` from 80 to 12.

## Decision tree (preconditions → action)

1. If DB error class is `connection acquire timeout` AND DB CPU is moderate → pool capacity, not compute.
2. If a recent DB deploy/config push exists → confirm by reading DB logs for the pool-shrink hint.
3. If a worker deploy *also* exists in the timeframe → rule it out explicitly by reading worker logs.
4. If confirmed → roll back the DB config, restart DB to pick up the restored pool.

## Action sequence (what to call, in order)

1. `query_logs(database)` — see "could not acquire connection" + the config-push deploy version.
2. `query_deploys(database)` — confirm DB deploy + the max_connections change.
3. `query_metrics(database, error_rate)` — confirm error class is pool-acquire, not compute.
4. `query_logs(worker)` — rule out the decoy worker deploy.
5. `submit_hypothesis(database_only_failure, [database, api-gateway, worker], 0.8, rollback_deploy)`.
6. `rollback_deploy(database)` — revert the offending config rollout.
7. `restart_service(database)` — restart against the restored pool config.
8. `run_check(database_recovery)` — verify DB pool health and write latency.
9. `run_check(end_to_end)` — verify gateway write-path succeeds.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Database `error_rate_pct` ≤ 2%; pool acquire timeouts disappear.
- Worker `error_rate_pct` returns to baseline (downstream pressure released).
- Gateway p95 latency drops to ≤ 100ms.
- Both checks pass; `incident_resolved == True`; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back the *worker* deploy is a wrong-target move and is penalized — worker code changes don't fix a DB config regression. Restarting the database *before* rolling back the config re-inherits the 12-connection pool and the symptom returns immediately.

---

## Run history

### Run 2026-04-23T22:01:33Z — Score 0.99 (10 steps)

Winning path:
1. `query_logs(database)` — Database is the loudest alert; inspect logs for the actual error signature
2. `query_deploys(database)` — Pool-acquire errors suggest a config change; check recent database rollouts
3. `query_metrics(database, error_rate)` — Confirm the error pattern is pool exhaustion rather than compute overload
4. `query_logs(worker)` — Rule out the decoy worker deploy by reading worker logs directly
5. `submit_hypothesis(database_only_failure)` — Localize the fault to the database config before remediating
6. `rollback_deploy(database)` — Roll back the offending database config rollout
7. `restart_service(database)` — Restart the database cleanly against the restored pool config
8. `run_check(database_recovery)` — Verify database pool health and write latency are back within SLO
9. `run_check(end_to_end)` — Verify gateway write-path traffic succeeds end-to-end
10. `declare_resolved` — Declare resolved only after objective checks pass

Checks passed: database_recovery, end_to_end.
