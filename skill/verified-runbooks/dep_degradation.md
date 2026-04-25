---
template_id: dep_degradation
status: draft
last_verified: null
---

# Cache Pool Exhaustion / Dependency Degradation

## Symptoms

Worker CPU is pinned at ~85% with rising connection-refused errors against the cache. Gateway latency p95 climbs to 400-500ms. Cache reports degraded with low CPU (~25%) but elevated error rate driven by `ERR max number of clients reached`. Database is healthy but its read load is *unusually* elevated because cache misses are diverting traffic to the DB direct path.

The trap: worker CPU is loud, so a junior responder will start investigating worker code or rolling back a worker deploy. The actual fault is one layer down — a recent cache deploy reduced `maxclients` from 1024 to 64 as a "cost optimization", and downstream consumers are now spinning on connection retries rather than doing real work.

## Decision tree (preconditions → action)

1. If gateway 5xx rate is high AND worker CPU is high AND database is healthy → suspect downstream-of-worker fault.
2. If cache `error_rate_pct` ≥ 15% with low CPU → connection-pool / capacity issue, not compute.
3. If cache deploy in the last hour AND error class is "connection rejected" → confirm by reading cache logs for the maxclients hint.
4. If confirmed → roll back the cache deploy, then restart cache to pick up restored config.

## Action sequence (what to call, in order)

1. `query_logs(worker)` — see the connection-refused pattern.
2. `query_logs(cache)` — see the "max number of clients reached" log.
3. `query_deploys(cache)` — confirm the maxclients-shrinking deploy and its timestamp.
4. `query_metrics(cache, error_rate)` — confirm error class, not capacity.
5. `submit_hypothesis(dependency_pool_exhausted, [cache, worker, api-gateway], 0.85, rollback_deploy)`.
6. `rollback_deploy(cache)` — revert the maxclients change.
7. `restart_service(cache)` — reload restored config.
8. `run_check(database_recovery)` — confirm DB read load returns to baseline.
9. `run_check(end_to_end)` — confirm session lookups round-trip through cache.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Cache `error_rate_pct` ≤ 1%; cache reports healthy.
- Worker CPU drops to baseline (~25%); error rate drops to ~0%.
- Database read volume returns to pre-incident levels.
- Both `database_recovery` and `end_to_end` checks pass.
- `incident_resolved == True` and `final_score` lands in [0.74, 0.80] for the scripted-optimal path.

## Rollback / safety notes

Restarting the cache *before* rolling back will reload the same 64-client cap and the symptom returns within seconds — premature restart is penalized by `failure_type=premature_restart`. Isolating the cache drops user traffic without addressing the root cause and inflates the blast-radius budget.
