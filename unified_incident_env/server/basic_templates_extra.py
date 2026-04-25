"""Round-2 Basic-tier templates rounding the catalogue out to 8 base + 4 hardened.

Each template here teaches a distinct SRE skill the original 6 do not cover:

- dep_degradation        — Redis/dependency pool exhaustion ("your service vs theirs")
- memory_leak_oom        — temporal pattern: restart count > error count
- auth_token_expiry      — cross-service credential propagation (cf. Vercel Apr 2026)
- network_partition      — connectivity inference when own metrics lie
- rate_limit_retry_storm — counterintuitive: more retries = more failure (cf. Stripe 2022)
- migration_lock         — lock contention without crash (cf. Railway Oct 2025)

All scenarios use the same 4-service topology (api-gateway / cache / database /
worker) the original Basic templates use, so they share the existing grader,
remediation engine, and procgen logic. Difficulty is tuned so that the deterministic
scripted-optimal baseline still tops out at ~0.70 and a noise-ignoring agent can
reach ~0.80 — leaving 0.20 of headroom for a trained policy.
"""

from __future__ import annotations

from typing import Any

from ..models import UnifiedIncidentAction
from .baselines import _ba

# Shared reward config — identical to existing templates so trajectory shaping is comparable.
_STD_REWARD = {
    "step_cost": 0.01,
    "redundant_action_penalty": 0.02,
    "unsafe_action_penalty": 0.08,
    "premature_resolution_penalty": 0.2,
    "successful_resolution_bonus": 0.25,
    "hypothesis_bonus_scale": 0.12,
    "forbidden_reward_sources": [
        "evidence_discovery",
        "query_success",
        "unlock_events",
        "stage_advancement",
        "patch_id_selection",
    ],
}

_HARD_REWARD = {
    **_STD_REWARD,
    "unsafe_action_penalty": 0.12,
    "premature_resolution_penalty": 0.3,
    "successful_resolution_bonus": 0.30,
}


EXTRA_TEMPLATES: dict[str, dict[str, Any]] = {
    "dep_degradation": {
        "id": "dep_degradation",
        "difficulty": "medium",
        "name": "Cache Pool Exhaustion / Dependency Degradation",
        "description": (
            "A cache (Redis) deploy lowered max_clients from 1024 to 64. The cache process is healthy "
            "in its own metrics, but new connections fail with 'max clients reached'. The worker — which "
            "depends on cache for session lookups — is degraded with high CPU as it spins on connection "
            "errors. Naïve responders blame the worker because its CPU is loud; the agent must follow the "
            "evidence to the cache deploy."
        ),
        "root_cause": "A bad cache deploy reduced the Redis max_clients limit and is starving downstream consumers.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.25,
            "database": 0.05,
            "api-gateway": 0.30,
            "cache": 0.40,
        },
        "reward_config": _STD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 41.0, "memory_pct": 36.0, "error_rate_pct": 18.0, "latency_ms": 480.0},
            "cache":       {"status": "degraded", "cpu_pct": 28.0, "memory_pct": 31.0, "error_rate_pct": 22.0, "latency_ms": 28.0},
            "database":    {"status": "healthy",  "cpu_pct": 27.0, "memory_pct": 33.0, "error_rate_pct": 1.0,  "latency_ms": 26.0},
            "worker":      {"status": "degraded", "cpu_pct": 84.0, "memory_pct": 52.0, "error_rate_pct": 14.0, "latency_ms": 360.0},
        },
        "initial_alerts": [
            {"service": "worker",      "severity": "critical", "message": "Worker CPU pinned at 84% with rising connection-refused errors against cache."},
            {"service": "api-gateway", "severity": "warning",  "message": "Login latency p95 climbed to 480ms; user impact climbing."},
            {"service": "cache",       "severity": "warning",  "message": "Cache rejecting new connections: 'max clients reached' since last rollout."},
        ],
        "logs": {
            "api-gateway": "Gateway upstream errors trace to worker session-lookup timeouts; no gateway code change in 24h.",
            "cache":       "Redis logs: 'ERR max number of clients reached' bursts since cache@2026.04.25-pool-shrink rollout dropped maxclients 1024 -> 64.",
            "database":    "Database is healthy and read-volume is actually elevated (clients bypass cache and hit DB direct).",
            "worker":      "Worker logs: 'redis: connection pool timeout' loops; CPU is consumed re-establishing failed connections, not real workload.",
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx rate at 18% — symptoms only; ratio matches worker session-lookup failure rate one-for-one.",
                "latency":    "Gateway p95 latency 480ms; pure waiting on worker.",
            },
            "cache": {
                "cpu":        "Cache CPU is moderate (28%); cache process itself is fine, the limit is connection-count not compute.",
                "error_rate": "22% of cache requests rejected at TCP-accept layer with 'max clients reached'.",
            },
            "database": {
                "cpu":        "Database CPU is elevated (~52%) — *unusually* elevated, because cache misses are diverting reads to DB.",
            },
            "worker": {
                "cpu":        "Worker CPU at 84% is busy-loop on connection retries, not real work.",
                "error_rate": "Worker error rate 14% — every error is a cache-acquire timeout, none originate from worker itself.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> cache (sessions) -> database (on cache miss)",
            "worker":      "worker -> cache (read-heavy) -> database (write-heavy)",
            "cache":       "cache fronts the sessions table; capacity is connection-count limited, not memory limited",
            "database":    "database is a passive participant; it sees elevated reads only because cache stopped serving them",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "Rolled out cache@2026.04.25-pool-shrink 14 minutes ago (maxclients 1024 -> 64; 'cost optimization').",
            "database":    "No database deploys in the last 24h.",
            "worker":      "No worker deploys in the last 24h.",
        },
        "checks": {
            "database_recovery": "Confirms database read load returns to baseline once cache resumes serving session lookups.",
            "end_to_end":        "Confirms a fresh login request resolves a session lookup through cache without falling back to DB.",
        },
        "truth": {
            "root_cause": "dependency_pool_exhausted",
            "affected_services": ["cache", "worker", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "cache",
            "restart_target": "cache",
            "isolate_target": "cache",
            "restart_requires_cause_removed": True,
            "incident_driver": "cache",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "cache": {"status": "degraded", "cpu_pct": 24.0, "memory_pct": 28.0, "error_rate_pct": 5.0, "latency_ms": 18.0},
        },
        "post_rollback_user_impact": 0.34,
        "post_rollback_slo_burn":    0.38,
        "post_restart_services": {
            "cache":       {"status": "healthy", "cpu_pct": 21.0, "memory_pct": 27.0, "error_rate_pct": 0.0, "latency_ms": 14.0},
            "worker":      {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 1.0, "latency_ms": 44.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 30.0, "error_rate_pct": 0.0, "latency_ms": 38.0},
            "database":    {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 32.0, "error_rate_pct": 0.0, "latency_ms": 22.0},
        },
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn":    0.14,
        "post_isolate_services": {
            "cache":       {"status": "isolated", "cpu_pct": 4.0, "memory_pct": 9.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "worker":      {"status": "degraded", "cpu_pct": 32.0, "memory_pct": 38.0, "error_rate_pct": 5.0, "latency_ms": 80.0},
            "database":    {"status": "degraded", "cpu_pct": 64.0, "memory_pct": 56.0, "error_rate_pct": 1.0, "latency_ms": 64.0},
        },
        "post_isolate_user_impact": 0.45,
        "post_isolate_slo_burn":    0.48,
        "degraded_services": {
            "cache":       {"status": "degraded", "cpu_pct": 28.0, "memory_pct": 31.0, "error_rate_pct": 22.0, "latency_ms": 28.0},
            "worker":      {"status": "degraded", "cpu_pct": 84.0, "memory_pct": 52.0, "error_rate_pct": 14.0, "latency_ms": 360.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 41.0, "memory_pct": 36.0, "error_rate_pct": 18.0, "latency_ms": 480.0},
        },
        "degraded_user_impact": 0.62,
        "degraded_slo_burn":    0.66,
        "failure_messages": {
            "wrong_rollback_target":  "Worker CPU is loud, but worker code is unchanged — rolling back worker won't fix a cache-side connection limit.",
            "low_value_restart":      "Restarting worker won't reset Redis maxclients; the limit is in the cache config.",
            "premature_restart":      "Restarting cache before rolling back the maxclients change will reload the same 64-client cap.",
            "wrong_isolation_target": "Isolating worker drops user traffic without addressing the cache pool limit.",
        },
        "difficulty_knobs": {
            "noise_services": ["sentry", "openai-proxy"],
            "noise_alerts": [
                {"service": "sentry",       "severity": "warning", "message": "Sentry release-health alert — downstream echo of worker errors, not a separate fault."},
                {"service": "openai-proxy", "severity": "warning", "message": "OpenAI proxy p95 latency +21ms (provider weather, unrelated)."},
            ],
            "noise_logs": {
                "sentry":       "Sentry alert is a downstream symptom of the worker error spike; not an independent fault.",
                "openai-proxy": "OpenAI proxy latency variation is provider-side and not on the session-lookup path.",
            },
            "blast_radius_budget": 2,
        },
    },
    "memory_leak_oom": {
        "id": "memory_leak_oom",
        "difficulty": "hard",
        "name": "Worker Memory Leak / OOM Restart Loop",
        "description": (
            "A recent worker deploy introduced a memory leak. The worker is in an OOM-kill restart "
            "loop: every ~90s it consumes its memory budget, gets killed, restarts, and connects "
            "back to the database — which sees a synchronization storm on every restart. The agent "
            "must recognize that *restart count* is climbing faster than *error rate*, and that the "
            "downstream database overload is reactive, not the cause."
        ),
        "root_cause": "A bad worker deploy is leaking memory and causing OOM restart loops that hammer the database on each restart.",
        "optimal_ticks": 11,
        "max_ticks": 13,
        "critical_service_weights": {
            "worker": 0.45,
            "database": 0.30,
            "api-gateway": 0.20,
            "cache": 0.05,
        },
        "reward_config": _HARD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 47.0, "memory_pct": 39.0, "error_rate_pct": 21.0, "latency_ms": 580.0},
            "cache":       {"status": "healthy",  "cpu_pct": 19.0, "memory_pct": 26.0, "error_rate_pct": 0.0,  "latency_ms": 14.0},
            "database":    {"status": "degraded", "cpu_pct": 78.0, "memory_pct": 64.0, "error_rate_pct": 11.0, "latency_ms": 410.0},
            "worker":      {"status": "crashed",  "cpu_pct": 12.0, "memory_pct": 96.0, "error_rate_pct": 8.0,  "latency_ms": 0.0},
        },
        "initial_alerts": [
            {"service": "worker",      "severity": "critical", "message": "Worker pod restart count: 14 in last 20 minutes (baseline 0). OOMKilled exit codes."},
            {"service": "database",    "severity": "warning",  "message": "Database connection-establish rate spiking every ~90s in lockstep with worker restarts."},
            {"service": "api-gateway", "severity": "warning",  "message": "Gateway 5xx error rate climbing as background jobs queue waiting on dead worker."},
        ],
        "logs": {
            "api-gateway": "Gateway logs: '502 upstream connect error' bursts; rate matches worker restart cadence one-for-one.",
            "cache":       "Cache is healthy and not on the worker memory path; cache hit rate is unchanged.",
            "database":    "Database logs: connection-establish bursts every ~90s, then quiet, then bursts again — consistent with caller restart loop, not a DB problem.",
            "worker":      "Worker logs: 'process killed (OOM)' every ~90s. Memory growth pattern began exactly at worker@2026.04.25-cache-prefetch rollout 35 minutes ago. Restart count: 14.",
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx rate is 21% with sawtooth pattern; bursts align 1:1 with worker pod restart timestamps.",
                "latency":    "Gateway p95 580ms — backlog draining each time worker comes back, then re-accumulating.",
            },
            "database": {
                "cpu":        "Database CPU 78% — but the load is connection-establish bursts, not query load.",
                "latency":    "Database write latency 410ms during connection-establish bursts; recovers between restarts.",
            },
            "worker": {
                "cpu":        "Worker CPU low (12%) — process spends most of its life dead, not working.",
                "error_rate": "Worker error rate (8%) is misleadingly low — most calls never reach worker because it's down.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database (write-heavy)",
            "worker":      "worker holds long-lived DB connections; restart causes re-establish storm",
            "database":    "database load is reactive — it's healthy when worker is healthy",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "No cache deploys in the last 24h.",
            "database":    "No database deploys in the last 24h.",
            "worker":      "Rolled out worker@2026.04.25-cache-prefetch 35 minutes ago (in-memory prefetch cache for hot keys; no per-request cache eviction).",
        },
        "checks": {
            "database_recovery": "Confirms database connection-establish rate returns to baseline once worker stops restart-looping.",
            "end_to_end":        "Confirms gateway -> worker -> DB request path completes without timing out.",
        },
        "truth": {
            "root_cause": "memory_leak_runaway",
            "affected_services": ["worker", "database", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "worker",
            "restart_target": "database",
            "isolate_target": "worker",
            "restart_requires_cause_removed": True,
            "incident_driver": "worker",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "worker": {"status": "healthy", "cpu_pct": 30.0, "memory_pct": 38.0, "error_rate_pct": 1.0, "latency_ms": 56.0},
        },
        "post_rollback_user_impact": 0.42,
        "post_rollback_slo_burn":    0.46,
        "post_restart_services": {
            "database":    {"status": "healthy", "cpu_pct": 36.0, "memory_pct": 42.0, "error_rate_pct": 0.0, "latency_ms": 28.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 30.0, "error_rate_pct": 0.0, "latency_ms": 36.0},
            "worker":      {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 36.0, "error_rate_pct": 1.0, "latency_ms": 48.0},
        },
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn":    0.14,
        "post_isolate_services": {
            "worker":   {"status": "isolated", "cpu_pct": 4.0,  "memory_pct": 12.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "database": {"status": "healthy",  "cpu_pct": 38.0, "memory_pct": 42.0, "error_rate_pct": 0.0, "latency_ms": 32.0},
        },
        "post_isolate_user_impact": 0.55,
        "post_isolate_slo_burn":    0.58,
        "degraded_services": {
            "worker":      {"status": "crashed",  "cpu_pct": 12.0, "memory_pct": 96.0, "error_rate_pct": 8.0,  "latency_ms": 0.0},
            "database":    {"status": "degraded", "cpu_pct": 78.0, "memory_pct": 64.0, "error_rate_pct": 11.0, "latency_ms": 410.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 47.0, "memory_pct": 39.0, "error_rate_pct": 21.0, "latency_ms": 580.0},
        },
        "degraded_user_impact": 0.78,
        "degraded_slo_burn":    0.84,
        "failure_messages": {
            "wrong_rollback_target":  "Database CPU is loud, but the load pattern is connection-establish bursts driven by worker restarts — not a DB-side fault.",
            "low_value_restart":      "Restarting database without first stopping worker just absorbs another restart-storm wave.",
            "premature_restart":      "Restarting database before rolling back worker means the next worker OOM will hammer it again.",
            "wrong_isolation_target": "Isolating gateway hides the symptom; worker will keep OOM-looping until the leaking deploy is rolled back.",
        },
        "difficulty_knobs": {
            "noise_services": ["sentry", "supabase-realtime"],
            "noise_alerts": [
                {"service": "sentry",             "severity": "warning",  "message": "Sentry release-health alert: new release error rate +120% (downstream echo of the actual fault)."},
                {"service": "supabase-realtime",  "severity": "warning",  "message": "Realtime subscription reconnect storm during scheduled credential rotation (unrelated)."},
            ],
            "noise_logs": {
                "sentry":            "Sentry alert is the downstream signal of the OOM loop; the root cause is worker memory not Sentry ingest.",
                "supabase-realtime": "Realtime reconnect storm is on a separate path and unrelated to worker memory growth.",
            },
            "blast_radius_budget": 2,
        },
    },
    "auth_token_expiry": {
        "id": "auth_token_expiry",
        "difficulty": "medium",
        "name": "Hardcoded Service Token Expired",
        "description": (
            "A worker deploy from yesterday hardcoded a service-account JWT against the auth provider. "
            "That JWT just expired. Worker calls to internal /v1/identity now return 401, which the "
            "worker propagates as 503 to upstream gateway. Gateway looks like the loud service in "
            "alerts, but it is a downstream symptom — the deploy regression is on worker, and rolling "
            "back the worker deploy restores credential refresh logic."
        ),
        "root_cause": "A worker deploy hardcoded a service-account token that has now expired, breaking credential refresh.",
        "optimal_ticks": 9,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.45,
            "database": 0.10,
            "api-gateway": 0.40,
            "cache": 0.05,
        },
        "reward_config": _STD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 35.0, "memory_pct": 38.0, "error_rate_pct": 27.0, "latency_ms": 220.0},
            "cache":       {"status": "healthy",  "cpu_pct": 18.0, "memory_pct": 24.0, "error_rate_pct": 0.0,  "latency_ms": 14.0},
            "database":    {"status": "healthy",  "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 1.0,  "latency_ms": 22.0},
            "worker":      {"status": "degraded", "cpu_pct": 31.0, "memory_pct": 38.0, "error_rate_pct": 34.0, "latency_ms": 95.0},
        },
        "initial_alerts": [
            {"service": "api-gateway", "severity": "critical", "message": "Gateway returning 503 on /v1/profile and /v1/billing endpoints (~27% error rate)."},
            {"service": "worker",      "severity": "warning",  "message": "Worker /v1/identity calls returning 401 from auth provider (~34% rate)."},
        ],
        "logs": {
            "api-gateway": "Gateway logs: '503 from upstream worker' on identity-touching paths. No gateway deploys in the last 24h.",
            "cache":       "Cache is unaffected; identity calls bypass cache.",
            "database":    "Database is healthy. The /v1/identity calls don't reach the DB — they fail at the auth provider.",
            "worker":      (
                "Worker logs: 'auth.provider returned 401 — token expired at 2026-04-25T12:00:00Z'. "
                "Token was hardcoded by worker@2026.04.24-identity-fix from yesterday; expiry was 24h."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 503 rate at 27% — concentrated exclusively on identity-touching endpoints.",
                "latency":    "Gateway latency on non-identity endpoints unchanged; only identity-path is regressing.",
            },
            "worker": {
                "error_rate": "Worker /v1/identity error rate 34% — every error is a 401 from the auth provider, none from worker logic.",
                "cpu":        "Worker CPU normal (31%); fault is auth not compute.",
            },
            "database": {
                "error_rate": "Database error rate near zero; identity-path failures never reach DB.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> auth-provider -> (identity operations)",
            "worker":      "worker holds a service-account credential to call auth-provider for token introspection",
            "database":    "database is off the identity path; not implicated",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "No cache deploys in the last 24h.",
            "database":    "No database deploys in the last 24h.",
            "worker":      (
                "Rolled out worker@2026.04.24-identity-fix 25 hours ago "
                "(hardcoded short-lived service-account JWT instead of using rotating secret store; expired 12 minutes ago)."
            ),
        },
        "checks": {
            "database_recovery": "Confirms the database identity-table read/write path stays healthy throughout (it never broke).",
            "end_to_end":        "Confirms a fresh authenticated user request resolves through the worker without 401.",
        },
        "truth": {
            "root_cause": "credential_rotation_breakage",
            "affected_services": ["worker", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "worker",
            "restart_target": "worker",
            "isolate_target": "worker",
            "restart_requires_cause_removed": True,
            "incident_driver": "worker",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "worker": {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 34.0, "error_rate_pct": 1.0, "latency_ms": 38.0},
        },
        "post_rollback_user_impact": 0.18,
        "post_rollback_slo_burn":    0.22,
        "post_restart_services": {
            "worker":      {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 0.0, "latency_ms": 32.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 24.0, "memory_pct": 28.0, "error_rate_pct": 0.0, "latency_ms": 30.0},
        },
        "post_restart_user_impact": 0.08,
        "post_restart_slo_burn":    0.12,
        "post_isolate_services": {
            "worker": {"status": "isolated", "cpu_pct": 4.0, "memory_pct": 12.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
        },
        "post_isolate_user_impact": 0.50,
        "post_isolate_slo_burn":    0.55,
        "degraded_services": {
            "worker":      {"status": "degraded", "cpu_pct": 31.0, "memory_pct": 38.0, "error_rate_pct": 34.0, "latency_ms": 95.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 35.0, "memory_pct": 38.0, "error_rate_pct": 27.0, "latency_ms": 220.0},
        },
        "degraded_user_impact": 0.55,
        "degraded_slo_burn":    0.60,
        "failure_messages": {
            "wrong_rollback_target":  "Gateway is the loudest service in alerts, but no gateway deploy in 24h — rolling it back doesn't refresh worker's expired token.",
            "low_value_restart":      "Restarting gateway doesn't reissue a worker token; the bad credential is hardcoded in worker code.",
            "premature_restart":      "Restarting worker before rolling back loads the same hardcoded expired token.",
            "wrong_isolation_target": "Isolating gateway drops healthy traffic without restoring identity calls.",
        },
        "difficulty_knobs": {
            "noise_services": ["clerk-auth", "stripe-webhook"],
            "noise_alerts": [
                {"service": "clerk-auth",     "severity": "warning", "message": "Clerk-auth public-JWKS fetch latency +18ms (unrelated provider weather)."},
                {"service": "stripe-webhook", "severity": "warning", "message": "Stripe webhook retry rate +6% (unrelated diurnal variance)."},
            ],
            "noise_logs": {
                "clerk-auth":     "Clerk JWKS latency variance is upstream provider weather; unrelated to internal token expiry.",
                "stripe-webhook": "Webhook retries are within normal bounds; not on the auth fault path.",
            },
            "blast_radius_budget": 1,
        },
    },
    "network_partition": {
        "id": "network_partition",
        "difficulty": "hard",
        "name": "Cache DNS Partition / Connectivity Inference",
        "description": (
            "A cache deploy changed the DNS bootstrap entry from cache.svc.cluster.local to "
            "10.42.7.4 — but that IP belongs to an evicted pod. Cache is healthy in its own metrics "
            "(it's running, accepting connections from its own ClusterIP), but every other service in "
            "the topology can't reach it. The agent must trust the connectivity evidence over the "
            "service's own self-report — which is the entire point of the lesson."
        ),
        "root_cause": "A bad cache deploy hardcoded an IP for service discovery that points at an evicted pod.",
        "optimal_ticks": 11,
        "max_ticks": 13,
        "critical_service_weights": {
            "worker": 0.30,
            "database": 0.05,
            "api-gateway": 0.30,
            "cache": 0.35,
        },
        "reward_config": _HARD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 39.0, "memory_pct": 36.0, "error_rate_pct": 19.0, "latency_ms": 510.0},
            "cache":       {"status": "healthy",  "cpu_pct": 14.0, "memory_pct": 22.0, "error_rate_pct": 0.0,  "latency_ms": 9.0},
            "database":    {"status": "healthy",  "cpu_pct": 27.0, "memory_pct": 33.0, "error_rate_pct": 1.0,  "latency_ms": 24.0},
            "worker":      {"status": "degraded", "cpu_pct": 24.0, "memory_pct": 36.0, "error_rate_pct": 31.0, "latency_ms": 4200.0},
        },
        "initial_alerts": [
            {"service": "worker",      "severity": "critical", "message": "Worker session-lookup p95 latency 4200ms; ~31% of session reads time out."},
            {"service": "api-gateway", "severity": "warning",  "message": "Gateway is queueing on worker; user-facing 5xx climbing."},
            {"service": "cache",       "severity": "warning",  "message": "Cache reports healthy from its own probes but inbound connection rate is near zero."},
        ],
        "logs": {
            "api-gateway": "Gateway upstream errors trace to worker 502s; no gateway deploys in 24h.",
            "cache":       (
                "Cache logs: process up, accepting connections on 6379, but inbound rate is 0.2/s vs baseline 1800/s. "
                "Last deploy cache@2026.04.25-dns-pin pinned the discovery target from svc DNS to a hardcoded pod IP."
            ),
            "database":    "Database is healthy; read volume is *up* because clients fall through to DB after cache lookups time out.",
            "worker":      (
                "Worker logs: 'dial tcp 10.42.7.4:6379 i/o timeout' every cache call. Pod IP 10.42.7.4 belongs to a "
                "pod evicted 38 minutes ago."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx 19% — symptoms only; latency tail is dominated by waiting on worker.",
                "latency":    "Gateway p95 510ms.",
            },
            "cache": {
                "cpu":        "Cache CPU 14% — process is fine; this is a connectivity problem not a compute problem.",
                "error_rate": "Cache error rate 0% — but inbound connection rate dropped 99.99% post-deploy.",
            },
            "database": {
                "cpu":        "Database CPU is *elevated* (47%) because cache misses fall through; this is a derivative effect.",
            },
            "worker": {
                "latency":    "Worker p95 latency 4200ms = exactly the cache-call timeout budget. Every slow request is the worker waiting on a TCP timeout.",
                "error_rate": "Worker error rate 31% — every error is a TCP i/o timeout to a single dead IP.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> cache -> database (only on miss)",
            "worker":      "worker resolves cache via DNS-or-hardcoded-IP; cache deploy changed that resolution",
            "cache":       "cache process is healthy in its own pod; reachability is determined by the discovery target the deploy hardcoded",
            "database":    "database is downstream and reactive",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "Rolled out cache@2026.04.25-dns-pin 38 minutes ago (pinned discovery from DNS svc to hardcoded pod IP 10.42.7.4 'to reduce DNS lookups').",
            "database":    "No database deploys in the last 24h.",
            "worker":      "No worker deploys in the last 24h.",
        },
        "checks": {
            "database_recovery": "Confirms database read load returns to baseline once cache is reachable.",
            "end_to_end":        "Confirms a fresh login resolves a session lookup through cache without falling through to DB.",
        },
        "truth": {
            "root_cause": "network_dns_partition",
            "affected_services": ["cache", "worker", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "cache",
            "restart_target": "cache",
            "isolate_target": "cache",
            "restart_requires_cause_removed": True,
            "incident_driver": "cache",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "cache": {"status": "degraded", "cpu_pct": 16.0, "memory_pct": 24.0, "error_rate_pct": 2.0, "latency_ms": 12.0},
        },
        "post_rollback_user_impact": 0.30,
        "post_rollback_slo_burn":    0.34,
        "post_restart_services": {
            "cache":       {"status": "healthy", "cpu_pct": 18.0, "memory_pct": 26.0, "error_rate_pct": 0.0, "latency_ms": 10.0},
            "worker":      {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 1.0, "latency_ms": 38.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 24.0, "memory_pct": 28.0, "error_rate_pct": 0.0, "latency_ms": 32.0},
            "database":    {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 32.0, "error_rate_pct": 0.0, "latency_ms": 22.0},
        },
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn":    0.14,
        "post_isolate_services": {
            "cache":  {"status": "isolated", "cpu_pct": 4.0, "memory_pct": 9.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "worker": {"status": "degraded", "cpu_pct": 30.0, "memory_pct": 36.0, "error_rate_pct": 5.0, "latency_ms": 110.0},
        },
        "post_isolate_user_impact": 0.46,
        "post_isolate_slo_burn":    0.50,
        "degraded_services": {
            "worker":      {"status": "degraded", "cpu_pct": 24.0, "memory_pct": 36.0, "error_rate_pct": 31.0, "latency_ms": 4200.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 39.0, "memory_pct": 36.0, "error_rate_pct": 19.0, "latency_ms": 510.0},
        },
        "degraded_user_impact": 0.66,
        "degraded_slo_burn":    0.72,
        "failure_messages": {
            "wrong_rollback_target":  "Cache reports healthy in its own metrics, but reachability is determined by the deploy that hardcoded the discovery IP — that's still the cache deploy.",
            "low_value_restart":      "Restarting worker doesn't change the hardcoded IP; worker will keep timing out on the same dead pod.",
            "premature_restart":      "Restarting cache before rolling back the discovery pin reloads the same dead-pod IP.",
            "wrong_isolation_target": "Isolating worker just drops user traffic; the cache discovery pin is still in place.",
        },
        "difficulty_knobs": {
            "noise_services": ["sessions-redis", "vercel-edge"],
            "noise_alerts": [
                {"service": "sessions-redis", "severity": "warning", "message": "sessions-redis latency +12ms — different cluster, unrelated."},
                {"service": "vercel-edge",    "severity": "warning", "message": "Vercel edge cache eviction event — frontend-only, unrelated."},
            ],
            "noise_logs": {
                "sessions-redis": "sessions-redis is on a different cluster and is unrelated to the internal cache reachability issue.",
                "vercel-edge":    "Vercel edge eviction touches static assets; not on the session-lookup path.",
            },
            "blast_radius_budget": 2,
        },
    },
    "rate_limit_retry_storm": {
        "id": "rate_limit_retry_storm",
        "difficulty": "hard",
        "name": "External Rate-Limit / Retry Storm",
        "description": (
            "Worker calls an external dependency (a payment processor / OpenAI API) and a recent "
            "deploy removed exponential backoff in favor of immediate fixed-interval retries. The "
            "dep started 429-throttling 30 minutes ago and the retries are amplifying the backlog. "
            "Database is degraded because every retry leaves an open transaction. The agent must "
            "recognize that *more retries = more failure* and that the deploy regressed retry policy."
        ),
        "root_cause": "A worker deploy removed exponential backoff and is amplifying an external rate-limit into a self-inflicted retry storm.",
        "optimal_ticks": 11,
        "max_ticks": 13,
        "critical_service_weights": {
            "worker": 0.40,
            "database": 0.30,
            "api-gateway": 0.25,
            "cache": 0.05,
        },
        "reward_config": _HARD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 44.0, "memory_pct": 38.0, "error_rate_pct": 24.0, "latency_ms": 720.0},
            "cache":       {"status": "healthy",  "cpu_pct": 18.0, "memory_pct": 24.0, "error_rate_pct": 0.0,  "latency_ms": 14.0},
            "database":    {"status": "degraded", "cpu_pct": 71.0, "memory_pct": 58.0, "error_rate_pct": 14.0, "latency_ms": 380.0},
            "worker":      {"status": "degraded", "cpu_pct": 92.0, "memory_pct": 64.0, "error_rate_pct": 42.0, "latency_ms": 5800.0},
        },
        "initial_alerts": [
            {"service": "worker",      "severity": "critical", "message": "Worker call rate to external dep is 30,000 RPM (baseline 600 RPM). 42% are returning 429."},
            {"service": "database",    "severity": "warning",  "message": "Database open-transaction count up 8x. Many transactions waiting on external dep callbacks."},
            {"service": "api-gateway", "severity": "warning",  "message": "Gateway p95 720ms — every request waiting on worker waiting on external dep."},
        ],
        "logs": {
            "api-gateway": "Gateway upstream errors point to worker timeouts; no gateway deploys in the last 24h.",
            "cache":       "Cache is unaffected; external dep calls don't go through cache.",
            "database":    (
                "Database logs: open transactions held for tens of seconds waiting on worker callbacks. "
                "Lock and connection pool pressure tracks worker call volume, not DB-side load."
            ),
            "worker":      (
                "Worker logs: '429 Too Many Requests' every ~50ms; retry count climbing without backoff. "
                "Retry policy was changed in worker@2026.04.25-no-backoff 36 minutes ago "
                "(removed exponential backoff in favor of fixed 50ms interval to 'reduce p99 tail latency')."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx 24% — derivative; matches worker request-timeout rate one-for-one.",
                "latency":    "Gateway p95 720ms — pure waiting on worker.",
            },
            "database": {
                "cpu":        "Database CPU 71% but the load is open-transaction count, not query volume — read/write QPS is actually *down*.",
                "error_rate": "Database error rate 14% — connection-pool exhaustion errors, not DB-engine errors.",
            },
            "worker": {
                "cpu":        "Worker CPU 92% spent retrying, not working — retry RPS is 50x baseline.",
                "error_rate": "Worker error rate 42% — every error is a 429 from external dep that the retry loop is amplifying.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> external-dep (rate-limited)",
            "worker":      "worker holds a DB transaction across each external-dep call; retries amplify both load axes",
            "database":    "database is reactive; transactions queue waiting on worker callbacks",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "No cache deploys in the last 24h.",
            "database":    "No database deploys in the last 24h.",
            "worker":      (
                "Rolled out worker@2026.04.25-no-backoff 36 minutes ago "
                "(replaced exp-backoff retry policy with fixed 50ms interval; commit message: 'reduce p99 tail latency')."
            ),
        },
        "checks": {
            "database_recovery": "Confirms database open-transaction count and connection-pool pressure return to baseline once retry storm subsides.",
            "end_to_end":        "Confirms gateway -> worker -> external-dep path completes with retries spaced sanely.",
        },
        "truth": {
            "root_cause": "external_rate_limit_storm",
            "affected_services": ["worker", "database", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "worker",
            "restart_target": "database",
            "isolate_target": "worker",
            "restart_requires_cause_removed": True,
            "incident_driver": "worker",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "worker": {"status": "degraded", "cpu_pct": 38.0, "memory_pct": 42.0, "error_rate_pct": 4.0, "latency_ms": 180.0},
        },
        "post_rollback_user_impact": 0.40,
        "post_rollback_slo_burn":    0.44,
        "post_restart_services": {
            "database":    {"status": "healthy", "cpu_pct": 32.0, "memory_pct": 38.0, "error_rate_pct": 0.0, "latency_ms": 28.0},
            "worker":      {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 1.0, "latency_ms": 60.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 24.0, "memory_pct": 28.0, "error_rate_pct": 0.0, "latency_ms": 38.0},
        },
        "post_restart_user_impact": 0.12,
        "post_restart_slo_burn":    0.14,
        "post_isolate_services": {
            "worker": {"status": "isolated", "cpu_pct": 4.0,  "memory_pct": 12.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "database": {"status": "healthy", "cpu_pct": 30.0, "memory_pct": 38.0, "error_rate_pct": 0.0, "latency_ms": 24.0},
        },
        "post_isolate_user_impact": 0.55,
        "post_isolate_slo_burn":    0.60,
        "degraded_services": {
            "worker":      {"status": "degraded", "cpu_pct": 92.0, "memory_pct": 64.0, "error_rate_pct": 42.0, "latency_ms": 5800.0},
            "database":    {"status": "degraded", "cpu_pct": 71.0, "memory_pct": 58.0, "error_rate_pct": 14.0, "latency_ms": 380.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 44.0, "memory_pct": 38.0, "error_rate_pct": 24.0, "latency_ms": 720.0},
        },
        "degraded_user_impact": 0.74,
        "degraded_slo_burn":    0.80,
        "failure_messages": {
            "wrong_rollback_target":  "Database CPU is loud, but the load is reactive (open transactions waiting on worker). Rolling back DB has no deploy to revert.",
            "low_value_restart":      "Restarting database without removing the retry storm just walks back into the same connection-pool exhaustion in seconds.",
            "premature_restart":      "Restarting database before rolling back worker means worker will hammer the new DB instance immediately.",
            "wrong_isolation_target": "Isolating gateway hides the symptom; worker keeps the retry loop running and DB keeps queuing.",
        },
        "difficulty_knobs": {
            "noise_services": ["openai-proxy", "sentry"],
            "noise_alerts": [
                {"service": "openai-proxy", "severity": "warning", "message": "OpenAI proxy 429 rate +400% — mostly *symptom* of the worker retry storm hitting upstream."},
                {"service": "sentry",       "severity": "warning", "message": "Sentry release-health alert: error rate +200% on latest worker release (downstream signal)."},
            ],
            "noise_logs": {
                "openai-proxy": "Proxy 429 rate is the upstream-side view of the worker storm; mitigating the storm fixes both.",
                "sentry":       "Sentry alert is downstream; the actual root cause is in the worker retry policy.",
            },
            "blast_radius_budget": 2,
        },
    },
    "migration_lock": {
        "id": "migration_lock",
        "difficulty": "medium",
        "name": "Database Migration Lock Contention",
        "description": (
            "A database deploy ran a CREATE INDEX without CONCURRENTLY on the orders table at peak "
            "traffic. The migration acquired an AccessExclusiveLock that all writes are now waiting "
            "behind. Database CPU is normal — this is a lock-contention failure, not a compute "
            "failure. The agent must recognize that lock waits, not query volume, are the primary "
            "signal, and roll back / cancel the migration deploy."
        ),
        "root_cause": "A database migration deploy is holding an AccessExclusiveLock on a busy table without CONCURRENTLY.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.20,
            "database": 0.50,
            "api-gateway": 0.25,
            "cache": 0.05,
        },
        "reward_config": _STD_REWARD,
        "initial_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 36.0, "memory_pct": 35.0, "error_rate_pct": 16.0, "latency_ms": 880.0},
            "cache":       {"status": "healthy",  "cpu_pct": 17.0, "memory_pct": 23.0, "error_rate_pct": 0.0,  "latency_ms": 13.0},
            "database":    {"status": "degraded", "cpu_pct": 22.0, "memory_pct": 41.0, "error_rate_pct": 33.0, "latency_ms": 7600.0},
            "worker":      {"status": "degraded", "cpu_pct": 18.0, "memory_pct": 28.0, "error_rate_pct": 28.0, "latency_ms": 6900.0},
        },
        "initial_alerts": [
            {"service": "database",    "severity": "critical", "message": "Database lock_wait_count at 412 (baseline 0). Write queries timing out at lock_timeout=8s."},
            {"service": "worker",      "severity": "warning",  "message": "Worker write-path errors 28% — every error is a database lock_timeout, not a worker fault."},
            {"service": "api-gateway", "severity": "warning",  "message": "Gateway p95 880ms; user-visible write requests timing out."},
        ],
        "logs": {
            "api-gateway": "Gateway upstream errors are downstream-driven; no gateway deploys in 24h.",
            "cache":       "Cache is healthy and unaffected; lock contention is on the orders table only.",
            "database":    (
                "Postgres logs: hundreds of 'process X waiting for AccessExclusiveLock on relation public.orders'. "
                "Holding session: pid 9128, query 'CREATE INDEX idx_orders_status ON public.orders(status)', "
                "started by db@2026.04.25-orders-index 18 minutes ago. CPU is *low* because no one can do work."
            ),
            "worker":      (
                "Worker logs: 'sql: lock_timeout 8000ms exceeded' on writes to orders table; "
                "reads from sessions/users tables are normal. Worker code is unchanged."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx 16% — concentrated on order-write paths only; reads unaffected.",
                "latency":    "Gateway p95 880ms — write paths queuing on DB; reads near-baseline.",
            },
            "database": {
                "cpu":        "Database CPU 22% (LOW) — the database isn't busy, it's locked. Lock contention is the signal, not CPU.",
                "error_rate": "Database lock_timeout error rate 33%; query-execution error rate is near zero (no actual SQL is broken).",
                "latency":    "Database p95 7600ms — but only on the locked table; reads of users/sessions are fast.",
            },
            "worker": {
                "error_rate": "Worker error rate 28% — every error is downstream lock_timeout; worker code is healthy.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database (write-heavy)",
            "worker":      "worker writes to orders table; lock contention blocks every write",
            "database":    "database is internally locked; AccessExclusiveLock on public.orders blocks all writers",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache":       "No cache deploys in the last 24h.",
            "database":    (
                "Applied db@2026.04.25-orders-index 18 minutes ago "
                "(CREATE INDEX idx_orders_status ON public.orders(status); ran *without* CONCURRENTLY at 14:00 UTC peak)."
            ),
            "worker":      "No worker deploys in the last 24h.",
        },
        "checks": {
            "database_recovery": "Confirms database lock_wait_count returns to 0 and write latency on orders table is back within SLO.",
            "end_to_end":        "Confirms a fresh order-write request succeeds end-to-end.",
        },
        "truth": {
            "root_cause": "migration_lock_contention",
            "affected_services": ["database", "worker", "api-gateway"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "database",
            "restart_target": "database",
            "isolate_target": "database",
            "restart_requires_cause_removed": True,
            "incident_driver": "database",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "database": {"status": "degraded", "cpu_pct": 36.0, "memory_pct": 42.0, "error_rate_pct": 4.0, "latency_ms": 80.0},
        },
        "post_rollback_user_impact": 0.30,
        "post_rollback_slo_burn":    0.34,
        "post_restart_services": {
            "database":    {"status": "healthy", "cpu_pct": 32.0, "memory_pct": 38.0, "error_rate_pct": 0.0, "latency_ms": 28.0},
            "worker":      {"status": "healthy", "cpu_pct": 22.0, "memory_pct": 30.0, "error_rate_pct": 1.0, "latency_ms": 50.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 24.0, "memory_pct": 28.0, "error_rate_pct": 0.0, "latency_ms": 36.0},
        },
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn":    0.14,
        "post_isolate_services": {
            "database": {"status": "isolated", "cpu_pct": 4.0,  "memory_pct": 12.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "worker":   {"status": "degraded", "cpu_pct": 14.0, "memory_pct": 26.0, "error_rate_pct": 6.0, "latency_ms": 100.0},
        },
        "post_isolate_user_impact": 0.60,
        "post_isolate_slo_burn":    0.66,
        "degraded_services": {
            "database":    {"status": "degraded", "cpu_pct": 22.0, "memory_pct": 41.0, "error_rate_pct": 33.0, "latency_ms": 7600.0},
            "worker":      {"status": "degraded", "cpu_pct": 18.0, "memory_pct": 28.0, "error_rate_pct": 28.0, "latency_ms": 6900.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 36.0, "memory_pct": 35.0, "error_rate_pct": 16.0, "latency_ms": 880.0},
        },
        "degraded_user_impact": 0.66,
        "degraded_slo_burn":    0.72,
        "failure_messages": {
            "wrong_rollback_target":  "Worker errors are loud, but worker code is unchanged in 24h — and rolling back worker doesn't release the AccessExclusiveLock on the orders table.",
            "low_value_restart":      "Restarting worker doesn't kill the holding migration session; lock_wait_count stays climbing.",
            "premature_restart":      "Restarting database before cancelling the migration deploy just lets it re-run on startup.",
            "wrong_isolation_target": "Isolating worker drops user write traffic; the lock contention is intrinsic to the database deploy.",
        },
        "difficulty_knobs": {
            "noise_services": ["analytics", "feature-flags"],
            "noise_alerts": [
                {"service": "analytics",     "severity": "warning", "message": "Analytics consumer lag +30s — unrelated reporting batch."},
                {"service": "feature-flags", "severity": "warning", "message": "Feature-flags subscriber reconnect — routine credential rotation."},
            ],
            "noise_logs": {
                "analytics":     "Analytics consumer lag tracks an unrelated batch job; not on the order-write path.",
                "feature-flags": "Feature-flags reconnect is a routine rotation; not relevant to the lock contention.",
            },
            "blast_radius_budget": 2,
        },
    },
}


def extra_baselines() -> dict[str, list[Any]]:
    """Scripted-optimal baselines for the round-2 templates.

    The shape mirrors challenge._BASELINE_BUILDERS so they integrate without
    touching the existing builder map.
    """
    return {
        "dep_degradation": lambda: [
            _ba("query_logs",        service="worker",     rationale="Worker is the loudest; check whether it's a worker-internal fault or downstream."),
            _ba("query_logs",        service="cache",      rationale="Worker errors point at cache acquire — read cache logs to confirm."),
            _ba("query_deploys",     service="cache",      rationale="A maxclients change in a recent cache deploy would explain the error pattern."),
            _ba("query_metrics",     service="cache", metric="error_rate", rationale="Confirm cache error class is connection-rejection, not cache-engine error."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "dependency_pool_exhausted",
                "affected_services": ["cache", "worker", "api-gateway"],
                "confidence": 0.85,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the cache deploy that lowered maxclients before remediating."),
            _ba("rollback_deploy",   service="cache",      rationale="Roll back the maxclients-shrinking cache deploy."),
            _ba("restart_service",   service="cache",      rationale="Restart cache to pick up the restored maxclients setting."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm DB read load returns to baseline."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm session lookups round-trip through cache."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
        "memory_leak_oom": lambda: [
            _ba("query_logs",        service="worker",     rationale="Worker restart count is the loudest signal; read logs for OOM evidence."),
            _ba("query_metrics",     service="worker", metric="cpu",     rationale="Confirm worker is NOT CPU-bound (which would suggest workload, not leak)."),
            _ba("query_deploys",     service="worker",     rationale="A recent worker deploy that introduced a long-lived buffer is the most likely cause."),
            _ba("query_metrics",     service="database", metric="cpu",   rationale="Confirm DB CPU spike pattern matches worker restart cadence (reactive, not primary)."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "memory_leak_runaway",
                "affected_services": ["worker", "database", "api-gateway"],
                "confidence": 0.85,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the leaking worker deploy before remediating."),
            _ba("rollback_deploy",   service="worker",     rationale="Roll back the leaking worker deploy."),
            _ba("restart_service",   service="database",   rationale="Restart DB cleanly so connection-establish backlog drains and re-stabilizes."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm DB connection-establish rate is back to baseline."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm gateway->worker->DB path completes."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
        "auth_token_expiry": lambda: [
            _ba("query_logs",        service="api-gateway", rationale="Gateway is the loudest in alerts; confirm whether it's gateway-internal or downstream."),
            _ba("query_logs",        service="worker",      rationale="Gateway errors trace to worker; read worker logs for the actual error class."),
            _ba("query_deploys",     service="worker",      rationale="A recent worker deploy that touched credential handling is the most likely cause."),
            _ba("query_deploys",     service="api-gateway", rationale="Rule out gateway deploy explicitly to dispel the noise."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "credential_rotation_breakage",
                "affected_services": ["worker", "api-gateway"],
                "confidence": 0.85,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the worker deploy that hardcoded the expiring token."),
            _ba("rollback_deploy",   service="worker",      rationale="Roll back the worker deploy that hardcoded the token; rolled-back code uses rotation logic."),
            _ba("restart_service",   service="worker",      rationale="Restart worker to pick up rotated credentials."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm gateway->worker->auth-provider succeeds."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm DB stayed healthy throughout."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
        "network_partition": lambda: [
            _ba("query_logs",        service="worker",     rationale="Worker p95 latency is the loudest; check what it's waiting on."),
            _ba("query_logs",        service="cache",      rationale="Worker is timing out on cache; read cache logs even though cache reports healthy."),
            _ba("query_deploys",     service="cache",      rationale="A cache deploy that changed discovery is the most likely cause."),
            _ba("query_dependencies", service="worker",    rationale="Confirm worker's reachability path to cache."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "network_dns_partition",
                "affected_services": ["cache", "worker", "api-gateway"],
                "confidence": 0.85,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the cache deploy that hardcoded the discovery IP."),
            _ba("rollback_deploy",   service="cache",      rationale="Roll back the discovery-pin cache deploy."),
            _ba("restart_service",   service="cache",      rationale="Restart cache so service discovery re-resolves through DNS."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm DB read load returns to baseline."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm session lookups round-trip through cache."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
        "rate_limit_retry_storm": lambda: [
            _ba("query_logs",        service="worker",      rationale="Worker error rate is the loudest; read logs for the 429 evidence and retry pattern."),
            _ba("query_metrics",     service="worker", metric="error_rate", rationale="Confirm error class is 429 from external dep, not worker-internal."),
            _ba("query_deploys",     service="worker",      rationale="A recent worker deploy that touched retry policy is the most likely cause."),
            _ba("query_metrics",     service="database", metric="cpu", rationale="Confirm DB CPU is moderate but lock/transaction count is what's high (reactive)."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "external_rate_limit_storm",
                "affected_services": ["worker", "database", "api-gateway"],
                "confidence": 0.85,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the worker deploy that removed exponential backoff."),
            _ba("rollback_deploy",   service="worker",      rationale="Roll back the no-backoff worker deploy."),
            _ba("restart_service",   service="database",    rationale="Restart DB so the open-transaction backlog drains."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm DB open-transaction count returns to baseline."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm gateway->worker->external-dep succeeds with sane retry spacing."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
        "migration_lock": lambda: [
            _ba("query_logs",        service="database",    rationale="Database has the loudest critical alert; read logs for the lock-wait pattern."),
            _ba("query_metrics",     service="database", metric="cpu",   rationale="Confirm CPU is low — the database is locked, not busy."),
            _ba("query_deploys",     service="database",    rationale="A recent migration deploy that ran without CONCURRENTLY is the most likely cause."),
            _ba("query_logs",        service="worker",      rationale="Confirm worker errors are downstream lock_timeouts, not a worker fault."),
            _ba("submit_hypothesis", hypothesis={
                "root_cause": "migration_lock_contention",
                "affected_services": ["database", "worker", "api-gateway"],
                "confidence": 0.88,
                "recommended_next_action": "rollback_deploy",
            }, rationale="Localize to the database migration deploy that holds the AccessExclusiveLock."),
            _ba("rollback_deploy",   service="database",    rationale="Roll back the non-concurrent migration deploy; cancel the holding session."),
            _ba("restart_service",   service="database",    rationale="Restart DB cleanly so any leftover lock state clears."),
            _ba("run_check",         check_name="database_recovery", rationale="Confirm lock_wait_count returns to 0."),
            _ba("run_check",         check_name="end_to_end",        rationale="Confirm a fresh order-write request succeeds."),
            _ba("declare_resolved", rationale="Declare resolved only after objective checks pass."),
        ],
    }
