# verified-runbooks/gateway_auth_rollout.md

Runbook entries are written by the sre-gym skill after a successful solve (incident_resolved=true and final_score > 0.85).
Each entry is immutable evidence — treat it as ground truth for the winning path.

---

## Run 2026-04-23T22:01:37Z — Score 0.99

- Steps: **8**
- Checks passed: database_recovery, end_to_end

**Winning path:**
1. `query_logs (service=api-gateway)` — Gateway is rejecting logins; read gateway logs to localize the rejection class
2. `query_deploys (service=api-gateway)` — Login rejection aligns with a recent auth middleware rollout; confirm deploy timing
3. `query_deploys (service=worker)` — Rule out the worker deploy explicitly rather than assuming
4. `submit_hypothesis (hypothesis=api_gateway_fault)` — Commit a calibrated hypothesis localizing to the gateway auth rollout
5. `rollback_deploy (service=api-gateway)` — Roll back the bad auth middleware rollout; no restart needed
6. `run_check (check_name=end_to_end)` — Verify that gateway login traffic now succeeds end-to-end
7. `run_check (check_name=database_recovery)` — Confirm the database is (and stayed) healthy throughout
8. `declare_resolved` — Declare resolved only after objective checks pass
