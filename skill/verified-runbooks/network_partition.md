---
template_id: network_partition
status: draft
last_verified: null
---

# Cache DNS Partition / Connectivity Inference

## Symptoms

Worker session-lookup p95 latency is in the multi-second range (4000ms+); ~30% of session reads time out. Gateway is queueing on worker; user-facing 5xx climbs steadily. Cache reports *healthy* in its own metrics — process is up, accepting connections on 6379, CPU is low (~14%) — but inbound connection rate has dropped near-zero.

The trap: cache says it's healthy. Self-reported health is misleading because reachability is determined by the discovery target, not the process state. A recent cache deploy hardcoded a pod IP for service discovery instead of using DNS, and that pod has since been evicted.

## Decision tree (preconditions → action)

1. If worker p95 latency = exactly the cache-call timeout budget → worker is waiting on a TCP timeout.
2. If cache's own metrics are healthy AND inbound rate dropped 99%+ → connectivity, not capacity.
3. If cache has a recent deploy that mentions "DNS", "discovery", "IP pin", or "reduce DNS lookups" → likely root cause.
4. If confirmed → roll back the cache discovery-pin deploy, restart cache so service discovery re-resolves.

## Action sequence (what to call, in order)

1. `query_logs(worker)` — see the `dial tcp ... i/o timeout` pattern.
2. `query_logs(cache)` — see the discovery-pin deploy log.
3. `query_deploys(cache)` — confirm the deploy that pinned discovery.
4. `query_dependencies(worker)` — confirm worker's reachability path to cache.
5. `submit_hypothesis(network_dns_partition, [cache, worker, api-gateway], 0.85, rollback_deploy)`.
6. `rollback_deploy(cache)` — revert the discovery-pin deploy.
7. `restart_service(cache)` — re-resolve discovery via DNS.
8. `run_check(database_recovery)` — confirm DB read load returns to baseline (cache-misses fall-through stops).
9. `run_check(end_to_end)` — confirm session lookups round-trip.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Worker `latency_ms` drops from 4000+ to ~40-60ms.
- Worker `error_rate_pct` returns to baseline (~1%).
- Cache inbound connection rate is back to ~1800/s (or whatever scenario baseline declares).
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back worker is a wrong-target move — worker code is unchanged in 24h. Restarting cache *before* rolling back reloads the same dead-pod IP. Isolating worker drops user traffic and the discovery pin remains in place.
