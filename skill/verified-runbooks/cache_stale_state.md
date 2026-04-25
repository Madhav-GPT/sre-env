---
template_id: cache_stale_state
status: draft
last_verified: null
---

# Cache TTL Regression / Stale State Leak

## Symptoms

Support ticket storm: ~14 users report seeing stale or *cross-user* session data. Cache hit ratio is *abnormally high* (98% vs baseline 72%) — that's the misleading signal that looks like a "win". Gateway 4xx rate up 9% on authenticated routes, but they're wrong-data symptoms not errors. Database read volume on the sessions table dropped 70% — cache is over-serving. Cache deploy in the last hour bumped session-key TTL from 30s to 3600s.

The trap: every standard reliability metric looks better than baseline (latency improved, hit ratio up, DB load down). But customer reports cross-tenant data leak. This is a containment-first scenario in *spirit* — the longer the bad TTL stays, the wider the cross-user data window grows.

## Decision tree (preconditions → action)

1. If cache hit ratio is *up* AND DB read volume is *down* AND customers report stale data → cache over-serving.
2. If cache deploy in the last hour mentions "TTL", "hit ratio", or "performance" → likely root cause.
3. If confirmed → roll back the cache deploy *then* restart cache to purge the poisoned entries.

## Action sequence (what to call, in order)

1. `query_logs(cache)` — see the TTL bump in the deploy log.
2. `query_deploys(cache)` — confirm the TTL-bumping deploy.
3. `query_metrics(database, cpu)` — confirm DB read volume drop (cache over-serving signal).
4. `submit_hypothesis(cache_ttl_regression, [cache, api-gateway], 0.85, rollback_deploy)`.
5. `rollback_deploy(cache)` — restore default TTL.
6. `restart_service(cache)` — purge the poisoned entries written under the bad TTL.
7. `run_check(end_to_end)` — confirm a fresh login presents that user's own data.
8. `run_check(database_recovery)` — confirm DB read volume returns to baseline.
9. `declare_resolved`.

## Success criteria (how you know you're done)

- Cache hit ratio returns to ~72% baseline.
- DB read volume on sessions table returns to baseline.
- No cross-user data reports in the next ~5 simulated minutes.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

**Restart is non-optional** for this scenario — without it the bad TTL entries stay cached for up to an hour, continuing the cross-user exposure window. Premature restart (before rolling back) just reloads the bad config. Isolating the cache forces fall-through to DB, which works but inflates the blast-radius budget and keeps the bad code path live.
