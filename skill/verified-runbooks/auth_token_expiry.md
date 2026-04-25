---
template_id: auth_token_expiry
status: draft
last_verified: null
---

# Hardcoded Service Token Expired

## Symptoms

Gateway returns 503 on identity-touching endpoints (~25-30% error rate). Worker shows 401s from the auth provider on the `/v1/identity` path. Database is healthy. Cache is healthy. The pattern is concentrated: only auth-touching routes regress; reads and writes that bypass identity checks are fine.

The trap: gateway is the loudest service in the alerts (it surfaces the customer-facing 503), and a junior responder will roll back gateway. But gateway has no recent deploy. Worker, on the other hand, deployed yesterday with a hardcoded service-account JWT whose 24-hour lifetime just expired.

## Decision tree (preconditions → action)

1. If gateway 503s are concentrated on identity routes only → suspect downstream auth.
2. If worker logs contain `401 from auth.provider` → the chain is gateway ← worker ← auth provider.
3. If worker has a recent deploy that mentions credential / identity / auth → likely root cause.
4. If gateway has *no* recent deploy → confirms worker is the regression source.
5. If confirmed → roll back the worker deploy (the rolled-back code uses rotation logic), then restart worker to pick up rotated credentials.

## Action sequence (what to call, in order)

1. `query_logs(api-gateway)` — confirm gateway is loud but downstream-driven.
2. `query_logs(worker)` — see the 401 pattern from the auth provider with the expired-token timestamp.
3. `query_deploys(worker)` — find the credential-handling deploy.
4. `query_deploys(api-gateway)` — rule out gateway explicitly.
5. `submit_hypothesis(credential_rotation_breakage, [worker, api-gateway], 0.85, rollback_deploy)`.
6. `rollback_deploy(worker)` — revert the deploy that hardcoded the token.
7. `restart_service(worker)` — pick up rotated credentials.
8. `run_check(end_to_end)` — confirm gateway → worker → auth-provider succeeds.
9. `run_check(database_recovery)` — confirm DB stayed healthy throughout.
10. `declare_resolved`.

## Success criteria (how you know you're done)

- Worker `error_rate_pct` ≤ 1%; gateway error rate returns to baseline.
- A fresh authenticated user request succeeds end-to-end.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back gateway is a wrong-target move (no gateway deploy to revert) — penalized with `failure_type=wrong_remediation_target`. Restarting worker without first rolling back loads the same hardcoded expired token and the symptom returns immediately.
