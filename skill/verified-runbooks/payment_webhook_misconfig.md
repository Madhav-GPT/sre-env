---
template_id: payment_webhook_misconfig
status: draft
last_verified: null
---

# Stripe Webhook Signature Regression

## Symptoms

Stripe dashboard shows ~47% of webhook deliveries failing signature verification since the last gateway deploy. Gateway 5xx is concentrated *only* on `/webhooks/stripe`; non-webhook endpoints are fine. Database writes to the subscriptions table dropped to near-zero in the last ~40 minutes. Support tickets: users report "paid but subscription inactive".

The trap: cache or worker might look "involved" via tangential alerts, but the payment path doesn't traverse them. Database is healthy (writes just aren't arriving). Stripe webhook retry queue is growing — that's a *symptom* of the upstream handler failure, not the cause.

## Decision tree (preconditions → action)

1. If gateway error rate is concentrated on a single path (e.g. `/webhooks/stripe`) → suspect that handler.
2. If a recent gateway deploy mentions "webhook", "stripe", or "signature" → likely root cause.
3. If DB stayed healthy AND error rate near zero → fault is upstream of DB.
4. If confirmed → roll back the gateway deploy; no restart needed for handler-class faults.

## Action sequence (what to call, in order)

1. `query_logs(api-gateway)` — see "Stripe signature verification failed" with the deploy version.
2. `query_deploys(api-gateway)` — confirm the webhook-handler-touching deploy.
3. `query_metrics(database, error_rate)` — confirm DB is *not* faulting; fault is upstream.
4. `submit_hypothesis(payment_webhook_regression, [api-gateway, database], 0.85, rollback_deploy)`.
5. `rollback_deploy(api-gateway)` — revert the bad webhook handler.
6. `run_check(end_to_end)` — confirm a fresh Stripe webhook round-trip succeeds.
7. `run_check(database_recovery)` — confirm subscriptions-table writes resume.
8. `declare_resolved`.

## Success criteria (how you know you're done)

- Gateway webhook-path error rate drops from 38%+ to ≤ 2%.
- Subscription writes resume in the database.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Restarting gateway without rolling back just brings the broken handler back up. Rolling back worker or cache doesn't fix a webhook regression on the gateway. Isolating gateway drops *all* user traffic, not just the webhook path — over-broad containment.
