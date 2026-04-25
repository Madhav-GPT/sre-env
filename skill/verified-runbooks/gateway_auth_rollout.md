---
template_id: gateway_auth_rollout
status: verified
last_verified: 2026-04-23T22:01:37Z
verified_score: 0.99
verified_steps: 8
---

# Gateway Auth Rollout Regression

## Symptoms

A new api-gateway auth-middleware rollout is rejecting ~40% of valid logins with 401. Gateway latency is *normal* (errors are fast rejections, not timeouts). Worker queue depth is elevated, but it's a downstream retry storm — worker code is unchanged. Database is healthy.

The trap: worker queue depth is loud and a recent worker deploy exists. But the worker deploy is a log-format tweak unrelated to auth; rolling back worker doesn't restore login traffic.

## Decision tree (preconditions → action)

1. If gateway error class is 401 (not timeouts) AND latency is normal → auth rejection, not capacity.
2. If a recent gateway deploy mentions auth, middleware, or token → likely root cause.
3. If worker errors are *retry-shaped* (high count, low local CPU) → worker is reacting to gateway 401s.
4. If confirmed → roll back the gateway deploy. No restart needed for middleware-class faults.

## Action sequence (what to call, in order)

1. `query_logs(api-gateway)` — see the auth-middleware rejection pattern + the deploy version.
2. `query_deploys(api-gateway)` — confirm the auth-touching deploy and timing.
3. `query_deploys(worker)` — explicitly rule out the worker deploy.
4. `submit_hypothesis(api_gateway_fault, [api-gateway, worker], 0.85, rollback_deploy)`.
5. `rollback_deploy(api-gateway)` — revert the bad auth middleware.
6. `run_check(end_to_end)` — verify login traffic succeeds.
7. `run_check(database_recovery)` — confirm DB is and stayed healthy throughout.
8. `declare_resolved`.

## Success criteria (how you know you're done)

- Gateway 401 rate drops from 40%+ to ≤ 1%.
- Worker queue depth returns to baseline.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Restarting gateway without rolling back loads the same broken middleware. Restarting worker doesn't fix the gateway auth regression. Isolating gateway drops *all* user traffic, not just auth — over-broad containment.

---

## Run history

### Run 2026-04-23T22:01:37Z — Score 0.99 (8 steps)

Winning path:
1. `query_logs(api-gateway)` — Gateway is rejecting logins; read gateway logs to localize the rejection class
2. `query_deploys(api-gateway)` — Login rejection aligns with a recent auth middleware rollout; confirm deploy timing
3. `query_deploys(worker)` — Rule out the worker deploy explicitly rather than assuming
4. `submit_hypothesis(api_gateway_fault)` — Commit a calibrated hypothesis localizing to the gateway auth rollout
5. `rollback_deploy(api-gateway)` — Roll back the bad auth middleware rollout; no restart needed
6. `run_check(end_to_end)` — Verify that gateway login traffic now succeeds end-to-end
7. `run_check(database_recovery)` — Confirm the database is (and stayed) healthy throughout
8. `declare_resolved` — Declare resolved only after objective checks pass

Checks passed: database_recovery, end_to_end.
