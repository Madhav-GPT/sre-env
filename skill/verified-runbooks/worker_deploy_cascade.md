# verified-runbooks/worker_deploy_cascade.md

Runbook entries are written by the sre-gym skill after a successful solve (incident_resolved=true and final_score > 0.85).
Each entry is immutable evidence — treat it as ground truth for the winning path.

---

## Run 2026-04-23T22:01:29Z — Score 0.99

- Steps: **10**
- Checks passed: database_recovery, end_to_end

**Winning path:**
1. `query_deploys (service=worker)` — Check whether any recent deploy aligns with the incident start
2. `query_logs (service=worker)` — Inspect worker logs because deploy timing and queue pressure suggest worker-originated harm
3. `query_metrics (service=database, metric=cpu)` — Confirm that the database is overloaded as a downstream effect
4. `query_dependencies (service=api-gateway)` — Verify the gateway depends on the worker and database path
5. `submit_hypothesis (hypothesis=bad_worker_deploy)` — Commit a calibrated hypothesis before taking an invasive mitigation step
6. `rollback_deploy (service=worker)` — Remove the triggering change before restarting downstream services
7. `restart_service (service=database)` — Bring the database back cleanly after the root cause is removed
8. `run_check (check_name=database_recovery)` — Verify the database is no longer crashing
9. `run_check (check_name=end_to_end)` — Verify gateway traffic succeeds end-to-end
10. `declare_resolved` — Declare resolved only after objective checks pass
