---
template_id: rate_limit_retry_storm
status: draft
last_verified: null
---

# External Rate-Limit / Retry Storm

## Symptoms

Worker call rate to an external dependency (payment processor / LLM proxy / etc.) is 50× baseline. ~40% of those calls return 429. Worker CPU is pinned at 92% — but spent retrying, not working. Database open-transaction count is 8× baseline; DB has connection-pool pressure but its query QPS is *down*. Gateway p95 latency is 700ms, propagating the worker delay.

The trap: DB CPU and connection-pool metrics look pathological, suggesting a database fault. But the load is *open transactions* held open across worker callbacks, not actual query volume. The actual root cause is a worker deploy that swapped exponential backoff for a fixed 50ms retry interval — every 429 from the external dep amplifies into 20 retries/second.

## Decision tree (preconditions → action)

1. If worker error rate is high AND error class is 429 → external rate-limit, not internal fault.
2. If DB connection-pool errors AND DB query QPS is *down* → reactive load, not primary.
3. If worker has a recent deploy mentioning "retry", "backoff", "p99", or "tail latency" → likely regression.
4. If confirmed → roll back the worker retry-policy deploy, then restart database to drain backlog.

## Action sequence (what to call, in order)

1. `query_logs(worker)` — see the 429 pattern + retry cadence.
2. `query_metrics(worker, error_rate)` — confirm error class is 429.
3. `query_deploys(worker)` — find the no-backoff deploy.
4. `query_metrics(database, cpu)` — confirm DB CPU is moderate; pressure is from open transactions.
5. `submit_hypothesis(external_rate_limit_storm, [worker, database, api-gateway], 0.85, rollback_deploy)`.
6. `rollback_deploy(worker)` — restore exponential-backoff policy.
7. `restart_service(database)` — drain the open-transaction backlog cleanly.
8. `run_check(database_recovery)` — confirm DB open-transaction count returns to baseline.
9. `run_check(end_to_end)` — confirm gateway → worker → external-dep succeeds with sane retry spacing.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Worker error rate drops from 40%+ to ~1%.
- DB open-transaction count returns to baseline.
- Gateway p95 latency settles below 100ms.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Restarting the database *before* rolling back the worker just pushes the connection-storm onto the freshly-restarted instance immediately. Isolating worker drops legitimate traffic — the storm is internal to the worker → external-dep loop, isolation doesn't quiet it.
