---
template_id: schema_drift_missing_migration
status: draft
last_verified: null
---

# Prisma Schema Drift / Missing Migration

## Symptoms

Sentry alert: `PrismaClientKnownRequestError: column users.plan_tier does not exist`. Gateway 500 rate is ~33% but concentrated on `/billing` and `/settings` only. Database is healthy. Worker plan-tier sync job fails with the same column-missing error. Postgres logs confirm the column genuinely isn't there; the migrations table's last applied entry is from a few days ago, before the schema change.

The trap: the database "passing" Postgres-level health checks lures you into thinking it's not the layer to investigate. It isn't — but the gateway *deploy* shipped Prisma client code expecting a column whose *migration* never ran in prod. Application/code shipped, schema didn't.

## Decision tree (preconditions → action)

1. If gateway 500s reference a specific column missing from DB → schema/code mismatch.
2. If a recent gateway deploy mentions schema, prisma, or migration → likely root cause.
3. If DB migrations table predates the schema change → migration didn't run.
4. If confirmed → roll back the gateway deploy to a schema-compatible version. Re-applying the migration is a separate workstream (ship-it-properly), not part of incident remediation.

## Action sequence (what to call, in order)

1. `query_logs(api-gateway)` — see the PrismaClientKnownRequestError.
2. `query_deploys(api-gateway)` — confirm the schema-expecting deploy.
3. `query_logs(database)` — verify the column genuinely isn't present.
4. `submit_hypothesis(schema_migration_mismatch, [api-gateway, worker, database], 0.88, rollback_deploy)`.
5. `rollback_deploy(api-gateway)` — revert to a schema-compatible client version.
6. `run_check(end_to_end)` — confirm `/billing` and `/settings` return 200 again.
7. `run_check(database_recovery)` — confirm DB stayed healthy throughout.
8. `declare_resolved`.

## Success criteria (how you know you're done)

- Gateway error rate on plan-tier-touching routes returns to baseline.
- Worker plan-tier sync job stops erroring.
- Both checks pass; `final_score` ≥ 0.74.

## Rollback / safety notes

Rolling back the database is wrong (no DB deploy to revert). Restarting gateway loads the same broken Prisma client. The proper post-incident workflow: ship the missing migration through CI with `prisma migrate deploy`, then redeploy the application code. That's separate from incident-window mitigation.
