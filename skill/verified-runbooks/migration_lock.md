---
template_id: migration_lock
status: draft
last_verified: null
---

# Database Migration Lock Contention

## Symptoms

Database `lock_wait_count` is in the hundreds (baseline 0). Write queries on the orders table time out at the configured `lock_timeout`. Worker reports 28% error rate — but every error is a downstream `lock_timeout`, not a worker fault. Database CPU is *low* (22%) — the database isn't busy, it's locked. Postgres logs show many sessions waiting for `AccessExclusiveLock on relation public.orders`, with the holding session running a `CREATE INDEX ... ON public.orders` query.

The trap: worker error volume is loud, suggesting a worker fault. But worker code is unchanged in 24h, and rolling back worker doesn't release the DB lock. The actual root cause is a recent database migration deploy that ran `CREATE INDEX` *without* `CONCURRENTLY` at peak traffic.

## Decision tree (preconditions → action)

1. If DB `lock_wait_count` is high AND DB CPU is low → contention, not compute.
2. If DB `error_rate_pct` is split between `lock_timeout` and zero engine errors → confirms lock pattern.
3. If a recent DB deploy mentions `CREATE INDEX`, `ALTER TABLE`, or `migration` → likely root cause.
4. If confirmed → roll back the DB migration deploy, restart DB to clear leftover lock state.

## Action sequence (what to call, in order)

1. `query_logs(database)` — see the AccessExclusiveLock waiters and the holding migration session.
2. `query_metrics(database, cpu)` — confirm CPU is *low*.
3. `query_deploys(database)` — find the migration deploy + confirm CONCURRENTLY was missing.
4. `query_logs(worker)` — confirm worker errors are downstream lock_timeouts.
5. `submit_hypothesis(migration_lock_contention, [database, worker, api-gateway], 0.88, rollback_deploy)`.
6. `rollback_deploy(database)` — cancel the holding migration session.
7. `restart_service(database)` — clear any leftover lock state.
8. `run_check(database_recovery)` — confirm `lock_wait_count` returns to 0.
9. `run_check(end_to_end)` — confirm a fresh order-write request succeeds.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- DB `lock_wait_count` is 0.
- DB write latency on the affected table returns to ≤ 50ms.
- Worker error rate drops to baseline (~1%).
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back worker is a wrong-target move — worker code is unchanged. Restarting DB *before* cancelling the migration deploy just lets the migration re-run on startup. Isolating DB drops *all* user traffic; lock contention is intrinsic to the deploy and isolation doesn't release it.

For the long-term fix, every `CREATE INDEX` migration should be `CREATE INDEX CONCURRENTLY` and run during off-peak windows; CI should reject migrations missing the keyword.
