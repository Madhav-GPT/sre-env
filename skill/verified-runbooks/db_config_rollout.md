# verified-runbooks/db_config_rollout.md

Runbook entries are written by the sre-gym skill after a successful solve (incident_resolved=true and final_score > 0.85).
Each entry is immutable evidence — treat it as ground truth for the winning path.

---

## Run 2026-04-23T22:01:33Z — Score 0.99

- Steps: **10**
- Checks passed: database_recovery, end_to_end

**Winning path:**
1. `query_logs (service=database)` — Database is the loudest alert; inspect logs for the actual error signature
2. `query_deploys (service=database)` — Pool-acquire errors suggest a config change; check recent database rollouts
3. `query_metrics (service=database, metric=error_rate)` — Confirm the error pattern is pool exhaustion rather than compute overload
4. `query_logs (service=worker)` — Rule out the decoy worker deploy by reading worker logs directly
5. `submit_hypothesis (hypothesis=database_only_failure)` — Localize the fault to the database config before remediating
6. `rollback_deploy (service=database)` — Roll back the offending database config rollout
7. `restart_service (service=database)` — Restart the database cleanly against the restored pool config
8. `run_check (check_name=database_recovery)` — Verify database pool health and write latency are back within SLO
9. `run_check (check_name=end_to_end)` — Verify gateway write-path traffic succeeds end-to-end
10. `declare_resolved` — Declare resolved only after objective checks pass
