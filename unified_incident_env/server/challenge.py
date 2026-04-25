"""Scenario catalog, baselines, and runtime helpers for the honest v2 core.

The catalogue ships 12 base templates: the original 6 (kept verbatim — they're
well-calibrated and shipped behavioural data) plus 6 round-2 templates added for
the April 2026 OpenEnv hackathon submission. See ``basic_templates_extra.py`` for
the new templates and ``docs/BASIC_TIER.md`` for the depth-not-quantity rationale.

Procedural generation expands each base into 5 variants (jittered metrics, deploy
timestamps, distractor noise rotation), giving 12 x 6 = 72 deterministic-but-
distinct scenarios at runtime. Holding out 12 (one per template) yields a clean
60-train / 12-eval split.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import random
import re
from typing import Any

from ..models import (
    BaselineCatalog,
    BaselineDefinition,
    BaselineStep,
    ScenarioCatalog,
    ScenarioSummary,
    UnifiedIncidentAction,
)
from .basic_templates_extra import EXTRA_TEMPLATES, extra_baselines

DEFAULT_SCENARIO_ID = "worker_deploy_cascade"
PROCGEN_VARIANTS_PER_TEMPLATE = 5
_MINUTES_AGO_RE = re.compile(r"(\d+)\s+minutes ago")
_ROLLOUT_VERSION_RE = re.compile(r"(@\d{4}\.\d{2}\.\d{2}-)([a-z0-9-]+)")

_BASE_SCENARIOS: dict[str, dict[str, Any]] = {
    "worker_deploy_cascade": {
        "id": "worker_deploy_cascade",
        "difficulty": "easy",
        "name": "Worker Deploy Cascade",
        "description": (
            "A bad worker deploy causes sustained database overload and login 502s at the gateway. "
            "The agent must diagnose from evidence, choose a safe remediation, verify recovery, and declare resolved only after checks pass."
        ),
        "root_cause": "A bad worker deploy is driving repeated database overload.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.4,
            "database": 0.4,
            "api-gateway": 0.2,
            "cache": 0.0,
        },
        "reward_config": {
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
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 61.0,
                "memory_pct": 38.0,
                "error_rate_pct": 24.0,
                "latency_ms": 640.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 18.0,
                "memory_pct": 24.0,
                "error_rate_pct": 0.0,
                "latency_ms": 14.0,
            },
            "database": {
                "status": "crashed",
                "cpu_pct": 99.0,
                "memory_pct": 97.0,
                "error_rate_pct": 100.0,
                "latency_ms": 0.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 88.0,
                "memory_pct": 71.0,
                "error_rate_pct": 19.0,
                "latency_ms": 420.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Login requests are returning sustained 502s.",
            },
            {
                "service": "database",
                "severity": "critical",
                "message": "Database process is crashing under repeated overload.",
            },
            {
                "service": "worker",
                "severity": "warning",
                "message": "Worker queue depth and retry volume spiked after a recent rollout.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway upstream errors point to worker timeouts followed by database connection failures. "
                "No recent gateway deploys are recorded."
            ),
            "cache": "Cache hit ratio is stable and cache upstream probes remain healthy.",
            "database": (
                "Database logs show repeated bursts of expensive worker-originated writes immediately before each crash."
            ),
            "worker": (
                "Worker logs show request fanout amplification and elevated retries beginning right after rollout build worker@2026.04.23-bad."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 502 rate is 24% and closely tracks worker timeout bursts.",
                "latency": "Gateway p95 latency climbed to 640ms while waiting on downstream worker/database calls.",
            },
            "database": {
                "cpu": "Database CPU is pinned at 99% until the process exits.",
                "latency": "Database latency spikes sharply before each crash loop.",
            },
            "worker": {
                "cpu": "Worker CPU is 88% with growing queue pressure.",
                "error_rate": "Worker retry/error rate is elevated after rollout.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database",
            "worker": "worker -> database",
            "database": "database is a terminal dependency for write-heavy worker jobs",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache": "No cache deploys in the last 24h.",
            "database": "No database deploys in the last 24h.",
            "worker": "Rolled out worker@2026.04.23-bad 12 minutes ago.",
        },
        "checks": {
            "database_recovery": "Confirms the database is healthy and no longer crashing.",
            "end_to_end": "Confirms login traffic succeeds without worker-induced overload.",
        },
        "truth": {
            "root_cause": "bad_worker_deploy",
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
            "worker": {"status": "healthy", "cpu_pct": 32.0, "memory_pct": 37.0, "error_rate_pct": 2.0, "latency_ms": 40.0},
        },
        "post_rollback_user_impact": 0.55,
        "post_rollback_slo_burn": 0.58,
        "post_restart_services": {
            "database": {"status": "healthy", "cpu_pct": 34.0, "memory_pct": 39.0, "error_rate_pct": 0.0, "latency_ms": 22.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 31.0, "error_rate_pct": 0.0, "latency_ms": 38.0},
        },
        "post_restart_user_impact": 0.14,
        "post_restart_slo_burn": 0.18,
        "post_isolate_services": {
            "worker": {"status": "isolated", "cpu_pct": 8.0, "memory_pct": 18.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "database": {"status": "healthy", "cpu_pct": 41.0, "memory_pct": 46.0, "error_rate_pct": 0.0, "latency_ms": 26.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 34.0, "memory_pct": 33.0, "error_rate_pct": 7.0, "latency_ms": 91.0},
        },
        "post_isolate_user_impact": 0.45,
        "post_isolate_slo_burn": 0.47,
        "degraded_services": {
            "worker": {"status": "degraded", "cpu_pct": 88.0, "memory_pct": 71.0, "error_rate_pct": 19.0, "latency_ms": 420.0},
            "database": {"status": "crashed", "cpu_pct": 99.0, "memory_pct": 97.0, "error_rate_pct": 100.0, "latency_ms": 0.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 61.0, "memory_pct": 38.0, "error_rate_pct": 24.0, "latency_ms": 640.0},
        },
        "degraded_user_impact": 0.82,
        "degraded_slo_burn": 0.91,
        "failure_messages": {
            "wrong_rollback_target": "Rolling back a service without a causal link wastes time and risk.",
            "low_value_restart": "Restarting that service is not the safe next remediation step for this incident.",
            "premature_restart": "Restarting before removing the trigger only causes another crash loop.",
            "wrong_isolation_target": "Isolating that service does not contain the dominant failure path.",
        },
        "difficulty_knobs": {
            "noise_services": ["stripe-webhook", "email-queue"],
            "noise_alerts": [
                {"service": "stripe-webhook", "severity": "warning", "message": "Stripe webhook retry volume slightly elevated (unrelated noise)."},
                {"service": "email-queue", "severity": "warning", "message": "Email queue depth up 15% on a recurring 6h cycle (unrelated noise)."},
            ],
            "noise_logs": {
                "stripe-webhook": "Webhook retries are within normal diurnal bounds; no payment-path regression.",
                "email-queue": "Queue depth tracks the usual Monday-evening marketing batch; no regression.",
            },
            "blast_radius_budget": 2,
        },
    },
    "db_config_rollout": {
        "id": "db_config_rollout",
        "difficulty": "medium",
        "name": "Database Config Rollout Regression",
        "description": (
            "A database config push cut connection pool size and write requests now time out. "
            "A separate worker deploy landed around the same time and looks suspicious but is not the cause. "
            "The agent must avoid the decoy, roll back the database config, restart it, and verify recovery."
        ),
        "root_cause": "A bad database config rollout shrank the connection pool and is dropping writes.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.2,
            "database": 0.5,
            "api-gateway": 0.3,
            "cache": 0.0,
        },
        "reward_config": {
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
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 44.0,
                "memory_pct": 36.0,
                "error_rate_pct": 17.0,
                "latency_ms": 520.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 20.0,
                "memory_pct": 26.0,
                "error_rate_pct": 0.0,
                "latency_ms": 15.0,
            },
            "database": {
                "status": "degraded",
                "cpu_pct": 62.0,
                "memory_pct": 54.0,
                "error_rate_pct": 48.0,
                "latency_ms": 880.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 51.0,
                "memory_pct": 44.0,
                "error_rate_pct": 12.0,
                "latency_ms": 310.0,
            },
        },
        "initial_alerts": [
            {
                "service": "database",
                "severity": "critical",
                "message": "Database connection acquire timeouts at 48% and climbing.",
            },
            {
                "service": "api-gateway",
                "severity": "warning",
                "message": "Write-path requests are returning sustained 5xx.",
            },
            {
                "service": "worker",
                "severity": "warning",
                "message": "Worker write latency is elevated; retries are climbing.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway upstream errors are downstream-driven: writes to the worker path return pool-exhaustion "
                "errors originating from the database. No gateway deploys recorded in the last 24h."
            ),
            "cache": "Cache reads are healthy and unrelated to the current write-path failures.",
            "database": (
                "Database logs show 'could not acquire connection' errors immediately after config rollout "
                "db@2026.04.24-cfg lowered max_connections from 80 to 12."
            ),
            "worker": (
                "Worker logs show retries driven by downstream database pool exhaustion, not local faults. "
                "Worker code deploy worker@2026.04.24-refactor is unrelated to the pool error signature."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 5xx rate is 17% and matches the database pool-exhaustion windows one-for-one.",
                "latency": "Gateway p95 climbed to 520ms waiting on database connection acquire.",
            },
            "database": {
                "cpu": "Database CPU is moderate (~62%), so this is not a compute overload pattern.",
                "error_rate": "Database error rate is 48% and dominated by 'connection acquire timeout'.",
                "latency": "Database write latency jumped to 880ms after the config rollout.",
            },
            "worker": {
                "cpu": "Worker CPU is 51% — no local overload; retries are reactive.",
                "error_rate": "Worker errors are retries against the saturated database pool.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database",
            "worker": "worker -> database",
            "database": "database is the terminal dependency; pool exhaustion here starves all upstream writers",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache": "No cache deploys in the last 24h.",
            "database": "Applied config db@2026.04.24-cfg 15 minutes ago (max_connections 80 -> 12).",
            "worker": "Rolled out worker@2026.04.24-refactor 22 minutes ago (unrelated code cleanup).",
        },
        "checks": {
            "database_recovery": "Confirms database write latency and pool health are back within SLO.",
            "end_to_end": "Confirms gateway write-path traffic succeeds end-to-end.",
        },
        "truth": {
            "root_cause": "database_only_failure",
            "affected_services": ["database", "api-gateway", "worker"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "database",
            "restart_target": "database",
            "isolate_target": None,
            "restart_requires_cause_removed": True,
            "incident_driver": "database",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "database": {"status": "degraded", "cpu_pct": 48.0, "memory_pct": 42.0, "error_rate_pct": 6.0, "latency_ms": 120.0},
        },
        "post_rollback_user_impact": 0.40,
        "post_rollback_slo_burn": 0.45,
        "post_restart_services": {
            "database": {"status": "healthy", "cpu_pct": 36.0, "memory_pct": 40.0, "error_rate_pct": 0.0, "latency_ms": 26.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 29.0, "memory_pct": 30.0, "error_rate_pct": 0.0, "latency_ms": 44.0},
            "worker": {"status": "healthy", "cpu_pct": 33.0, "memory_pct": 36.0, "error_rate_pct": 1.0, "latency_ms": 48.0},
        },
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn": 0.14,
        "post_isolate_services": {},
        "post_isolate_user_impact": 0.70,
        "post_isolate_slo_burn": 0.75,
        "degraded_services": {
            "database": {"status": "degraded", "cpu_pct": 62.0, "memory_pct": 54.0, "error_rate_pct": 48.0, "latency_ms": 880.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 44.0, "memory_pct": 36.0, "error_rate_pct": 17.0, "latency_ms": 520.0},
            "worker": {"status": "degraded", "cpu_pct": 51.0, "memory_pct": 44.0, "error_rate_pct": 12.0, "latency_ms": 310.0},
        },
        "degraded_user_impact": 0.70,
        "degraded_slo_burn": 0.78,
        "failure_messages": {
            "wrong_rollback_target": "The worker deploy is a decoy; worker errors are reactive to database pool exhaustion.",
            "low_value_restart": "Restarting that service does not address a database-config regression.",
            "premature_restart": "Restarting the database before rolling back the config will re-inherit the 12-connection pool and fail again.",
            "wrong_isolation_target": "Isolation is not useful here: the cause is a config regression, not a runaway service.",
        },
        "difficulty_knobs": {
            "noise_services": ["sessions-redis", "analytics"],
            "noise_alerts": [
                {"service": "sessions-redis", "severity": "warning", "message": "Sessions-redis p99 latency nudged up 8ms (unrelated noise)."},
                {"service": "analytics", "severity": "warning", "message": "Analytics consumer lag up to 45s from baseline 30s (unrelated noise)."},
            ],
            "noise_logs": {
                "sessions-redis": "No errors on sessions-redis; hit ratio stable.",
                "analytics": "Analytics consumer lag fluctuation consistent with upstream Kafka producer batching, unrelated to current incident.",
            },
            "blast_radius_budget": 2,
        },
    },
    "gateway_auth_rollout": {
        "id": "gateway_auth_rollout",
        "difficulty": "hard",
        "name": "Gateway Auth Rollout Regression",
        "description": (
            "A new api-gateway auth-middleware rollout is rejecting ~40% of valid logins. "
            "A recent worker deploy and elevated worker queue depth make the worker look like a plausible suspect. "
            "The agent must localize to the gateway, roll back its deploy, and verify recovery without unnecessary restarts."
        ),
        "root_cause": "A bad api-gateway auth-middleware rollout is rejecting valid logins.",
        "optimal_ticks": 8,
        "max_ticks": 10,
        "critical_service_weights": {
            "worker": 0.15,
            "database": 0.15,
            "api-gateway": 0.70,
            "cache": 0.0,
        },
        "reward_config": {
            "step_cost": 0.01,
            "redundant_action_penalty": 0.02,
            "unsafe_action_penalty": 0.12,
            "premature_resolution_penalty": 0.3,
            "successful_resolution_bonus": 0.3,
            "hypothesis_bonus_scale": 0.12,
            "forbidden_reward_sources": [
                "evidence_discovery",
                "query_success",
                "unlock_events",
                "stage_advancement",
                "patch_id_selection",
            ],
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 38.0,
                "memory_pct": 42.0,
                "error_rate_pct": 41.0,
                "latency_ms": 180.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 17.0,
                "memory_pct": 23.0,
                "error_rate_pct": 0.0,
                "latency_ms": 12.0,
            },
            "database": {
                "status": "healthy",
                "cpu_pct": 38.0,
                "memory_pct": 41.0,
                "error_rate_pct": 1.0,
                "latency_ms": 28.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 63.0,
                "memory_pct": 48.0,
                "error_rate_pct": 4.0,
                "latency_ms": 220.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Gateway is returning 401 on ~40% of valid login attempts.",
            },
            {
                "service": "worker",
                "severity": "warning",
                "message": "Worker queue depth is elevated from the retry storm upstream.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway logs show auth-middleware rejecting tokens with valid signatures. "
                "Rejection rate started exactly at the gateway@2026.04.24-auth rollout boundary."
            ),
            "cache": "Cache hit ratio stable and unrelated.",
            "database": "Database logs are clean; no increase in errors or latency.",
            "worker": (
                "Worker logs show client-side retry storms triggered by upstream 401s, not local faults. "
                "Worker deploy worker@2026.04.24-hotfix is a log-format tweak and does not touch auth."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway error rate is 41%, dominated by 401 responses (auth failures).",
                "latency": "Gateway latency is normal — errors are fast rejections, not timeouts.",
            },
            "database": {
                "cpu": "Database CPU is 38% (normal).",
                "error_rate": "Database error rate is ~1% and flat.",
            },
            "worker": {
                "cpu": "Worker CPU is 63% from retry volume, not workload.",
                "error_rate": "Worker errors are reactive retries, not primary failures.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> (auth) -> worker -> database",
            "worker": "worker -> database",
            "database": "database is healthy; it is not on the fault path",
        },
        "deploy_history": {
            "api-gateway": "Rolled out gateway@2026.04.24-auth 9 minutes ago (auth middleware rewrite).",
            "cache": "No cache deploys in the last 24h.",
            "database": "No database deploys in the last 24h.",
            "worker": "Rolled out worker@2026.04.24-hotfix 18 minutes ago (log-format tweak, no auth changes).",
        },
        "checks": {
            "database_recovery": "Confirms the database is healthy (always healthy in this scenario).",
            "end_to_end": "Confirms gateway login traffic succeeds end-to-end.",
        },
        "truth": {
            "root_cause": "api_gateway_fault",
            "affected_services": ["api-gateway", "worker"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "api-gateway",
            "restart_target": None,
            "isolate_target": "api-gateway",
            "restart_requires_cause_removed": True,
            "incident_driver": "api-gateway",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "api-gateway": {"status": "healthy", "cpu_pct": 30.0, "memory_pct": 34.0, "error_rate_pct": 1.0, "latency_ms": 38.0},
            "worker": {"status": "healthy", "cpu_pct": 34.0, "memory_pct": 36.0, "error_rate_pct": 1.0, "latency_ms": 52.0},
        },
        "post_rollback_user_impact": 0.12,
        "post_rollback_slo_burn": 0.18,
        "post_restart_services": {},
        "post_restart_user_impact": 0.12,
        "post_restart_slo_burn": 0.18,
        "post_isolate_services": {
            "api-gateway": {"status": "isolated", "cpu_pct": 6.0, "memory_pct": 14.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
        },
        "post_isolate_user_impact": 0.55,
        "post_isolate_slo_burn": 0.60,
        "degraded_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 38.0, "memory_pct": 42.0, "error_rate_pct": 41.0, "latency_ms": 180.0},
            "worker": {"status": "degraded", "cpu_pct": 63.0, "memory_pct": 48.0, "error_rate_pct": 4.0, "latency_ms": 220.0},
        },
        "degraded_user_impact": 0.65,
        "degraded_slo_burn": 0.72,
        "failure_messages": {
            "wrong_rollback_target": "The worker deploy is a log-format tweak and is not on the auth fault path.",
            "low_value_restart": "Restarting a service does not fix a config/middleware regression rolled out as a deploy.",
            "premature_restart": "Restarting before rolling back the gateway auth change just restarts the same bad middleware.",
            "wrong_isolation_target": "Isolating workers or database cuts healthy traffic without fixing the gateway auth fault.",
        },
        "difficulty_knobs": {
            "noise_services": ["stripe-webhook", "image-cdn", "feature-flags"],
            "noise_alerts": [
                {"service": "stripe-webhook", "severity": "warning", "message": "Stripe webhook signing drift warning — known benign noise from clock skew."},
                {"service": "image-cdn", "severity": "warning", "message": "Image CDN purge lag on asia-east1 edge (unrelated noise)."},
                {"service": "feature-flags", "severity": "warning", "message": "Feature-flags subscriber reconnected after routine rotation (unrelated noise)."},
            ],
            "noise_logs": {
                "stripe-webhook": "Webhook signature log shows no delivery failures; flagged warnings are clock-skew benign.",
                "image-cdn": "CDN purge lag is within published SLA; no customer-visible impact.",
                "feature-flags": "Feature-flags consumer reconnect logs are routine rotation; no delivery loss.",
            },
            "blast_radius_budget": 1,
        },
    },
    "payment_webhook_misconfig": {
        "id": "payment_webhook_misconfig",
        "difficulty": "medium",
        "name": "Stripe Webhook Signature Regression",
        "description": (
            "A recent api-gateway deploy shipped a webhook handler with the wrong Stripe API "
            "version and signature-verification logic. ~40% of webhooks now fail signature check. "
            "Users are paying but subscriptions never activate. Support tickets are climbing. "
            "The agent must localize the fault to the gateway deploy (not Stripe itself), roll back, "
            "and verify end-to-end recovery."
        ),
        "root_cause": "A bad api-gateway deploy broke Stripe webhook signature verification.",
        "optimal_ticks": 9,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.1,
            "database": 0.3,
            "api-gateway": 0.6,
            "cache": 0.0,
        },
        "reward_config": {
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
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 34.0,
                "memory_pct": 39.0,
                "error_rate_pct": 38.0,
                "latency_ms": 210.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 19.0,
                "memory_pct": 25.0,
                "error_rate_pct": 0.0,
                "latency_ms": 13.0,
            },
            "database": {
                "status": "healthy",
                "cpu_pct": 31.0,
                "memory_pct": 37.0,
                "error_rate_pct": 1.0,
                "latency_ms": 28.0,
            },
            "worker": {
                "status": "healthy",
                "cpu_pct": 27.0,
                "memory_pct": 34.0,
                "error_rate_pct": 2.0,
                "latency_ms": 42.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Stripe dashboard shows 47% of webhook deliveries failing signature verification since last deploy.",
            },
            {
                "service": "api-gateway",
                "severity": "warning",
                "message": "Support: 12 users report 'paid but subscription inactive' in the last 30 minutes.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway webhook handler logs: 'Stripe signature verification failed: no signatures found matching the expected signature'. "
                "Trace points at gateway@2026.04.24-stripe-fix rollout from the webhook handler rewrite."
            ),
            "cache": "Cache is idle on the webhook path; payment flow does not touch cache.",
            "database": (
                "Database writes to the subscriptions table dropped to near-zero in the last 40 minutes. "
                "No schema or query-plan changes recorded."
            ),
            "worker": "Worker queue is not involved in the webhook payment path; no anomaly.",
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Webhook-path error rate is 38% and matches the Stripe delivery-failure rate exactly.",
                "latency": "Non-webhook gateway latency is unchanged; only /webhooks/stripe is regressing.",
            },
            "database": {
                "error_rate": "Database error rate is steady near zero; the fault is upstream of the DB.",
                "latency": "Database latency unchanged.",
            },
            "worker": {
                "error_rate": "Worker error rate is baseline; worker is not on the payment write path.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> stripe webhook verification -> database (subscriptions table write)",
            "worker": "worker is not part of the payment webhook path",
            "database": "database receives subscription writes only from the gateway webhook handler",
        },
        "deploy_history": {
            "api-gateway": "Rolled out gateway@2026.04.24-stripe-fix 18 minutes ago (Stripe webhook handler rewrite; API version bump 2023-10-16 -> 2024-06-20).",
            "cache": "No cache deploys in the last 24h.",
            "database": "No database deploys in the last 24h.",
            "worker": "No worker deploys in the last 24h.",
        },
        "checks": {
            "database_recovery": "Confirms the subscriptions-table write path is restored.",
            "end_to_end": "Confirms a fresh Stripe webhook round-trip succeeds end-to-end.",
        },
        "truth": {
            "root_cause": "payment_webhook_regression",
            "affected_services": ["api-gateway", "database"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "api-gateway",
            "restart_target": None,
            "isolate_target": "api-gateway",
            "restart_requires_cause_removed": True,
            "incident_driver": "api-gateway",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "api-gateway": {"status": "healthy", "cpu_pct": 28.0, "memory_pct": 33.0, "error_rate_pct": 1.0, "latency_ms": 36.0},
        },
        "post_rollback_user_impact": 0.12,
        "post_rollback_slo_burn": 0.15,
        "post_restart_services": {},
        "post_restart_user_impact": 0.12,
        "post_restart_slo_burn": 0.15,
        "post_isolate_services": {
            "api-gateway": {"status": "isolated", "cpu_pct": 5.0, "memory_pct": 12.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
        },
        "post_isolate_user_impact": 0.62,
        "post_isolate_slo_burn": 0.68,
        "degraded_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 34.0, "memory_pct": 39.0, "error_rate_pct": 38.0, "latency_ms": 210.0},
        },
        "degraded_user_impact": 0.45,
        "degraded_slo_burn": 0.52,
        "failure_messages": {
            "wrong_rollback_target": "Rolling back worker or cache doesn't fix a webhook handler regression on the gateway.",
            "low_value_restart": "Restarting doesn't undo the bad signature-verification code that shipped in the gateway deploy.",
            "premature_restart": "Restarting before rolling back the gateway deploy just brings the broken handler back up.",
            "wrong_isolation_target": "Isolating the wrong service drops healthy traffic without containing the webhook failure.",
        },
        "difficulty_knobs": {
            "noise_services": ["stripe-webhook", "sentry", "analytics"],
            "noise_alerts": [
                {"service": "stripe-webhook", "severity": "warning", "message": "Stripe webhook retry queue growing — symptom of upstream handler failures (not a separate fault)."},
                {"service": "sentry", "severity": "warning", "message": "Sentry ingest latency +18ms — unrelated diurnal pattern."},
                {"service": "analytics", "severity": "warning", "message": "Analytics consumer lag +22s from baseline — unrelated batch job."},
            ],
            "noise_logs": {
                "stripe-webhook": "Stripe dashboard shows delivery failures tracking gateway error rate 1:1 — stripe-webhook is a downstream symptom, not the cause.",
                "sentry": "Sentry ingest delay is a known weekly pattern during their batch index rebuild.",
                "analytics": "Analytics consumer lag tracks end-of-quarter reporting load and is independent of the payment path.",
            },
            "blast_radius_budget": 1,
        },
    },
    "schema_drift_missing_migration": {
        "id": "schema_drift_missing_migration",
        "difficulty": "medium",
        "name": "Prisma Schema Drift / Missing Migration",
        "description": (
            "A new api-gateway deploy ships Prisma client code expecting a `users.plan_tier` column "
            "that was added to the schema file but whose migration was never applied to production. "
            "Any /billing or /settings route that references plan_tier now 500s. The agent must "
            "recognize that the database is healthy but the application-side schema expectation is wrong, "
            "roll back the gateway deploy, and verify recovery."
        ),
        "root_cause": "A bad api-gateway deploy assumes a database column whose migration never shipped.",
        "optimal_ticks": 9,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.2,
            "database": 0.2,
            "api-gateway": 0.6,
            "cache": 0.0,
        },
        "reward_config": {
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
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 41.0,
                "memory_pct": 44.0,
                "error_rate_pct": 33.0,
                "latency_ms": 260.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 18.0,
                "memory_pct": 24.0,
                "error_rate_pct": 0.0,
                "latency_ms": 14.0,
            },
            "database": {
                "status": "healthy",
                "cpu_pct": 29.0,
                "memory_pct": 36.0,
                "error_rate_pct": 2.0,
                "latency_ms": 24.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 38.0,
                "memory_pct": 41.0,
                "error_rate_pct": 11.0,
                "latency_ms": 180.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Sentry: PrismaClientKnownRequestError 'column users.plan_tier does not exist' — 33% error rate on /billing.",
            },
            {
                "service": "worker",
                "severity": "warning",
                "message": "Worker plan-tier sync job failing with the same column-missing error from the ORM layer.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway logs: 'PrismaClientKnownRequestError: column `users.plan_tier` does not exist in the current database' "
                "beginning right after gateway@2026.04.24-plan-tiers was rolled out."
            ),
            "cache": "Cache is unaffected; plan-tier lookups never reach the cache because the query fails at the ORM layer.",
            "database": (
                "Database is healthy. Postgres logs confirm the column genuinely doesn't exist; "
                "migrations table shows last applied migration was 2026-04-18, before the plan_tier schema change."
            ),
            "worker": (
                "Worker logs show the same column-missing errors from its plan-tier sync job; "
                "no worker deploy today, the worker just happens to share the Prisma client."
            ),
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 500 rate is 33% and concentrated exclusively on /billing and /settings routes.",
                "latency": "Gateway latency on unaffected routes is normal; only plan-tier-touching endpoints are broken.",
            },
            "database": {
                "error_rate": "Database query error rate is near zero — queries fail at the ORM, never reach the DB.",
            },
            "worker": {
                "error_rate": "Worker sync-job error rate tracks the gateway exactly; same Prisma client version, same fault.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> prisma client -> database (column query rejected at ORM level)",
            "worker": "worker also depends on the same prisma client; both broke together because they share a code artifact",
            "database": "database is a passive participant; it rejects the query because the column truly isn't there",
        },
        "deploy_history": {
            "api-gateway": "Rolled out gateway@2026.04.24-plan-tiers 22 minutes ago (schema update + Prisma client regenerate; migration was generated but the apply step was skipped in CI).",
            "cache": "No cache deploys in the last 24h.",
            "database": "No database schema changes applied in the last 72h (migrations table last entry: 2026-04-18T14:02Z).",
            "worker": "No worker deploys in the last 24h; worker inherits the new Prisma client via shared package.",
        },
        "checks": {
            "database_recovery": "Confirms the database schema and ORM queries are consistent.",
            "end_to_end": "Confirms /billing and /settings routes return 200 again.",
        },
        "truth": {
            "root_cause": "schema_migration_mismatch",
            "affected_services": ["api-gateway", "worker", "database"],
            "best_next_action": "rollback_deploy",
        },
        "remediation_recipe": {
            "rollback_target": "api-gateway",
            "restart_target": None,
            "isolate_target": "api-gateway",
            "restart_requires_cause_removed": True,
            "incident_driver": "api-gateway",
            "resolution_check": "end_to_end",
        },
        "post_rollback_services": {
            "api-gateway": {"status": "healthy", "cpu_pct": 26.0, "memory_pct": 32.0, "error_rate_pct": 1.0, "latency_ms": 34.0},
            "worker": {"status": "healthy", "cpu_pct": 24.0, "memory_pct": 30.0, "error_rate_pct": 1.0, "latency_ms": 44.0},
        },
        "post_rollback_user_impact": 0.10,
        "post_rollback_slo_burn": 0.14,
        "post_restart_services": {},
        "post_restart_user_impact": 0.10,
        "post_restart_slo_burn": 0.14,
        "post_isolate_services": {
            "api-gateway": {"status": "isolated", "cpu_pct": 4.0, "memory_pct": 10.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
        },
        "post_isolate_user_impact": 0.58,
        "post_isolate_slo_burn": 0.66,
        "degraded_services": {
            "api-gateway": {"status": "degraded", "cpu_pct": 41.0, "memory_pct": 44.0, "error_rate_pct": 33.0, "latency_ms": 260.0},
            "worker": {"status": "degraded", "cpu_pct": 38.0, "memory_pct": 41.0, "error_rate_pct": 11.0, "latency_ms": 180.0},
        },
        "degraded_user_impact": 0.48,
        "degraded_slo_burn": 0.55,
        "failure_messages": {
            "wrong_rollback_target": "Rolling back the database or worker doesn't undo the gateway's schema-expecting deploy.",
            "low_value_restart": "Restarting won't create the missing column; the gateway code must be rolled back to a schema-compatible version.",
            "premature_restart": "Restarting before rolling back just brings the same broken Prisma client back up.",
            "wrong_isolation_target": "Isolating healthy services drops traffic without fixing the ORM-layer mismatch.",
        },
        "difficulty_knobs": {
            "noise_services": ["supabase-realtime", "sentry", "feature-flags"],
            "noise_alerts": [
                {"service": "supabase-realtime", "severity": "warning", "message": "Supabase-realtime subscription reconnect storm during routine credential rotation (unrelated)."},
                {"service": "sentry", "severity": "warning", "message": "Sentry release-health warning: new release has >1% regression — expected given the deploy correlation."},
                {"service": "feature-flags", "severity": "warning", "message": "Feature-flags config push for an experiment unrelated to the billing path."},
            ],
            "noise_logs": {
                "supabase-realtime": "Supabase-realtime reconnect chatter is independent of the billing write path.",
                "sentry": "Sentry release-health alert is a downstream echo of the gateway error rate, not a separate fault.",
                "feature-flags": "Feature-flags rollout targets onboarding funnel only; no billing-path gating.",
            },
            "blast_radius_budget": 1,
        },
    },
    "cache_stale_state": {
        "id": "cache_stale_state",
        "difficulty": "medium",
        "name": "Cache TTL Regression / Stale State Leak",
        "description": (
            "A cache deploy bumped session-object TTL from 30s to 3600s. Users now see stale data, "
            "and a few support tickets report seeing other users' display names on page refresh. "
            "Cache metrics look healthy (hit ratio is up!) which is the trap. The agent must "
            "localize the fault to the cache deploy, roll back, restart to purge the poisoned state, "
            "and verify recovery."
        ),
        "root_cause": "A bad cache deploy increased session TTL and is serving stale, sometimes cross-user, state.",
        "optimal_ticks": 10,
        "max_ticks": 12,
        "critical_service_weights": {
            "worker": 0.1,
            "database": 0.1,
            "api-gateway": 0.3,
            "cache": 0.5,
        },
        "reward_config": {
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
        },
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 32.0,
                "memory_pct": 38.0,
                "error_rate_pct": 9.0,
                "latency_ms": 82.0,
            },
            "cache": {
                "status": "degraded",
                "cpu_pct": 21.0,
                "memory_pct": 28.0,
                "error_rate_pct": 0.0,
                "latency_ms": 16.0,
            },
            "database": {
                "status": "healthy",
                "cpu_pct": 26.0,
                "memory_pct": 34.0,
                "error_rate_pct": 0.0,
                "latency_ms": 22.0,
            },
            "worker": {
                "status": "healthy",
                "cpu_pct": 24.0,
                "memory_pct": 31.0,
                "error_rate_pct": 1.0,
                "latency_ms": 48.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "Support ticket storm: 14 users report seeing stale or cross-user session data in the last hour.",
            },
            {
                "service": "cache",
                "severity": "warning",
                "message": "Cache hit ratio is abnormally high (98% vs baseline 72%) — consistent with over-long TTL.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Gateway logs: authenticated user_id in session does not match database user_id on "
                "~3% of requests — ratio matches the inverse of session turnover. No gateway deploy in 24h."
            ),
            "cache": (
                "Cache logs show bumped default TTL from 30s to 3600s via cache@2026.04.24-ttl-perf rollout. "
                "Engineer note in the deploy: 'Improves cache hit ratio; should cut DB read cost.' "
                "Session keys were not excluded from the TTL bump."
            ),
            "database": (
                "Database read volume on sessions table dropped 70% in the last 45 minutes — consistent with "
                "cache over-serving, not with an app-level bug."
            ),
            "worker": "Worker is not on the session read/write path; no anomaly.",
        },
        "metrics": {
            "api-gateway": {
                "error_rate": "Gateway 4xx rate is +9% on authenticated routes — not errors per se, but wrong-data symptoms.",
                "latency": "Gateway latency is actually improved (cache over-serving), which is the misleading signal.",
            },
            "cache": {
                "latency": "Cache latency is baseline; cache is technically healthy by its own metrics.",
            },
            "database": {
                "cpu": "Database CPU dropped as cache started over-serving — looks like a win until you see the user reports.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> cache (session lookup) -> database (on cache miss)",
            "cache": "cache sits in front of sessions table; TTL controls cross-user isolation window",
            "database": "database is passive; cache is masking state divergence from the DB",
        },
        "deploy_history": {
            "api-gateway": "No gateway deploys in the last 24h.",
            "cache": "Rolled out cache@2026.04.24-ttl-perf 35 minutes ago (default TTL 30s -> 3600s for 'hit ratio'; no per-key overrides).",
            "database": "No database deploys in the last 72h.",
            "worker": "No worker deploys in the last 24h.",
        },
        "checks": {
            "database_recovery": "Confirms the sessions table read volume and cache consistency window are back to baseline.",
            "end_to_end": "Confirms a fresh login presents that user's own data on refresh, not a stale snapshot.",
        },
        "truth": {
            "root_cause": "cache_ttl_regression",
            "affected_services": ["cache", "api-gateway"],
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
            "cache": {"status": "healthy", "cpu_pct": 20.0, "memory_pct": 27.0, "error_rate_pct": 0.0, "latency_ms": 14.0},
        },
        "post_rollback_user_impact": 0.22,
        "post_rollback_slo_burn": 0.24,
        "post_restart_services": {
            "cache": {"status": "healthy", "cpu_pct": 18.0, "memory_pct": 25.0, "error_rate_pct": 0.0, "latency_ms": 13.0},
            "api-gateway": {"status": "healthy", "cpu_pct": 27.0, "memory_pct": 32.0, "error_rate_pct": 1.0, "latency_ms": 38.0},
        },
        "post_restart_user_impact": 0.08,
        "post_restart_slo_burn": 0.12,
        "post_isolate_services": {
            "cache": {"status": "isolated", "cpu_pct": 3.0, "memory_pct": 9.0, "error_rate_pct": 0.0, "latency_ms": 0.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 41.0, "memory_pct": 46.0, "error_rate_pct": 4.0, "latency_ms": 120.0},
            "database": {"status": "degraded", "cpu_pct": 58.0, "memory_pct": 52.0, "error_rate_pct": 2.0, "latency_ms": 44.0},
        },
        "post_isolate_user_impact": 0.40,
        "post_isolate_slo_burn": 0.44,
        "degraded_services": {
            "cache": {"status": "degraded", "cpu_pct": 21.0, "memory_pct": 28.0, "error_rate_pct": 0.0, "latency_ms": 16.0},
            "api-gateway": {"status": "degraded", "cpu_pct": 32.0, "memory_pct": 38.0, "error_rate_pct": 9.0, "latency_ms": 82.0},
        },
        "degraded_user_impact": 0.58,
        "degraded_slo_burn": 0.60,
        "failure_messages": {
            "wrong_rollback_target": "Rolling back gateway/worker/db doesn't flush stale entries from a cache deployed with bad TTL.",
            "low_value_restart": "Restarting a service that didn't introduce bad TTL doesn't purge the poisoned cache state.",
            "premature_restart": "Restarting the cache before rolling back just reintroduces the bad TTL config.",
            "wrong_isolation_target": "Isolating a healthy service drops user traffic without flushing the stale cache state.",
        },
        "difficulty_knobs": {
            "noise_services": ["stripe-webhook", "openai-proxy", "clerk-auth"],
            "noise_alerts": [
                {"service": "stripe-webhook", "severity": "warning", "message": "Stripe webhook retry rate +4% — within normal diurnal variance."},
                {"service": "openai-proxy", "severity": "warning", "message": "OpenAI proxy p95 latency +32ms — upstream provider weather, unrelated to sessions."},
                {"service": "clerk-auth", "severity": "warning", "message": "Clerk-auth JWKS cache refresh ran 18 minutes late — unrelated to session cache."},
            ],
            "noise_logs": {
                "stripe-webhook": "Webhook retry volume is within normal bounds; no session involvement.",
                "openai-proxy": "OpenAI proxy latency reflects provider-side weather and is not on the session-data path.",
                "clerk-auth": "Clerk JWKS refresh delay is within SLA and is independent of the internal session cache.",
            },
            "blast_radius_budget": 2,
        },
    },
}


def _stable_rng(*parts: object) -> random.Random:
    seed_material = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _jitter_metric(value: float, *, rng: random.Random, spread: float, floor: float = 0.0, ceil: float = 100.0) -> float:
    if value == 0.0:
        return 0.0
    delta = value * rng.uniform(-spread, spread)
    return round(_clamp(value + delta, floor, ceil), 1)


def _jitter_latency(value: float, *, rng: random.Random, spread: float) -> float:
    if value == 0.0:
        return 0.0
    delta = value * rng.uniform(-spread, spread)
    return round(max(0.0, value + delta), 1)


def _mutate_service_table(table: dict[str, dict[str, Any]], *, rng: random.Random, spread: float) -> dict[str, dict[str, Any]]:
    mutated: dict[str, dict[str, Any]] = {}
    for service_name, payload in table.items():
        item = dict(payload)
        item["cpu_pct"] = _jitter_metric(float(item["cpu_pct"]), rng=rng, spread=spread)
        item["memory_pct"] = _jitter_metric(float(item["memory_pct"]), rng=rng, spread=spread)
        item["error_rate_pct"] = _jitter_metric(float(item["error_rate_pct"]), rng=rng, spread=spread)
        item["latency_ms"] = _jitter_latency(float(item["latency_ms"]), rng=rng, spread=spread)
        mutated[service_name] = item
    return mutated


def _mutate_deploy_text(text: str, *, rng: random.Random, service: str) -> str:
    age_minutes = rng.randint(6, 28)
    rollout_suffix = f"{service[:3]}{rng.randint(11, 98)}"
    updated = _MINUTES_AGO_RE.sub(f"{age_minutes} minutes ago", text, count=1)
    return _ROLLOUT_VERSION_RE.sub(rf"\1{rollout_suffix}", updated, count=1)


def _mutate_noise_knobs(knobs: dict[str, Any], *, rng: random.Random, variant_index: int) -> dict[str, Any]:
    mutated = deepcopy(knobs)
    noise_services = list(mutated.get("noise_services", []))
    if not noise_services:
        return mutated
    rotation = variant_index % len(noise_services)
    rotated_services = noise_services[rotation:] + noise_services[:rotation]
    alert_pool = {item["service"]: dict(item) for item in mutated.get("noise_alerts", [])}
    log_pool = dict(mutated.get("noise_logs", {}))
    selected_count = min(len(rotated_services), max(1, 1 + (variant_index % len(rotated_services))))
    selected_services = rotated_services[:selected_count]
    mutated["noise_services"] = selected_services
    mutated["noise_alerts"] = [alert_pool[service] for service in selected_services if service in alert_pool]
    mutated["noise_logs"] = {service: log_pool[service] for service in selected_services if service in log_pool}
    return mutated


def _procgen_variant_id(template_id: str, variant_index: int) -> str:
    return f"{template_id}__p{variant_index + 1:02d}"


def _materialize_procgen_variant(template_id: str, template: dict[str, Any], *, variant_index: int) -> dict[str, Any]:
    rng = _stable_rng(template_id, variant_index)
    spread_by_difficulty = {
        "easy": 0.05,
        "medium": 0.08,
        "hard": 0.10,
    }
    spread = spread_by_difficulty.get(template["difficulty"], 0.06)
    scenario = deepcopy(template)
    scenario["id"] = _procgen_variant_id(template_id, variant_index)
    scenario["template_id"] = template_id
    scenario["is_procgen"] = True
    scenario["name"] = f"{template['name']} [procgen {variant_index + 1}]"
    scenario["description"] = (
        f"{template['description']} "
        f"Variant {variant_index + 1} reshuffles timing and distractor noise."
    )
    for key in (
        "initial_services",
        "degraded_services",
        "post_rollback_services",
        "post_restart_services",
        "post_isolate_services",
    ):
        scenario[key] = _mutate_service_table(template.get(key, {}), rng=rng, spread=spread)
    scenario["deploy_history"] = {
        service: _mutate_deploy_text(text, rng=rng, service=service)
        for service, text in template.get("deploy_history", {}).items()
    }
    scenario["difficulty_knobs"] = _mutate_noise_knobs(template.get("difficulty_knobs", {}), rng=rng, variant_index=variant_index)
    return scenario


def _build_scenarios() -> dict[str, dict[str, Any]]:
    # Merge round-2 Basic templates into the canonical base catalogue. We do
    # this inside _build_scenarios (rather than at module import time) so that
    # any future tier-aware filter can short-circuit easily without mutating
    # module state.
    base = {**_BASE_SCENARIOS, **EXTRA_TEMPLATES}
    catalog: dict[str, dict[str, Any]] = {}
    for template_id, scenario in base.items():
        catalog[template_id] = deepcopy(scenario)
        catalog[template_id]["template_id"] = template_id
        catalog[template_id]["is_procgen"] = False
        for variant_index in range(PROCGEN_VARIANTS_PER_TEMPLATE):
            variant = _materialize_procgen_variant(
                template_id,
                catalog[template_id],
                variant_index=variant_index,
            )
            catalog[variant["id"]] = variant
    return catalog


SCENARIOS: dict[str, dict[str, Any]] = _build_scenarios()

_RUNTIME_PROGRESS: dict[str, Any] | None = None


def get_scenario(scenario_id: str) -> dict[str, Any]:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id {scenario_id!r}")
    return deepcopy(SCENARIOS[scenario_id])


SUPPORTED_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


def scenario_for_difficulty(difficulty: str, seed: int | None = None) -> dict[str, Any]:
    matches = [
        scenario
        for scenario in SCENARIOS.values()
        if scenario["difficulty"] == difficulty
    ]
    if seed is None:
        for scenario in matches:
            if not scenario.get("is_procgen", False):
                return deepcopy(scenario)
    if matches:
        return deepcopy(matches[(seed or 0) % len(matches)])
    raise ValueError(f"Unknown difficulty {difficulty!r}")


def list_scenarios(difficulty: str | None = None, include_procgen: bool = True) -> ScenarioCatalog:
    if difficulty is not None and difficulty not in SUPPORTED_DIFFICULTIES:
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    scenarios = [
        ScenarioSummary(
            id=scenario["id"],
            difficulty=scenario["difficulty"],
            name=scenario["name"],
            description=scenario["description"],
            root_cause=scenario["root_cause"],
            optimal_ticks=scenario["optimal_ticks"],
        )
        for scenario in SCENARIOS.values()
        if (difficulty is None or scenario["difficulty"] == difficulty)
        and (include_procgen or not scenario.get("is_procgen", False))
    ]
    return ScenarioCatalog(
        default_scenario_id=DEFAULT_SCENARIO_ID,
        available_difficulties=list(SUPPORTED_DIFFICULTIES),
        filtered_difficulty=difficulty,
        scenarios=scenarios,
    )


def _worker_cascade_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="worker"),
            rationale="Check whether any recent deploy aligns with the incident start.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="worker"),
            rationale="Inspect worker logs because deploy timing and queue pressure suggest worker-originated harm.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_metrics", service="database", metric="cpu"),
            rationale="Confirm that the database is overloaded as a downstream effect.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_dependencies", service="api-gateway"),
            rationale="Verify the gateway depends on the worker and database path.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "bad_worker_deploy",
                    "affected_services": ["worker", "database", "api-gateway"],
                    "confidence": 0.82,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Commit a calibrated hypothesis before taking an invasive mitigation step.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="worker"),
            rationale="Remove the triggering change before restarting downstream services.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="restart_service", service="database"),
            rationale="Bring the database back cleanly after the root cause is removed.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Verify the database is no longer crashing.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify gateway traffic succeeds end-to-end.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def _db_config_rollout_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="database"),
            rationale="Database is the loudest alert; inspect logs for the actual error signature.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="database"),
            rationale="Pool-acquire errors suggest a config change; check recent database rollouts.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_metrics", service="database", metric="error_rate"),
            rationale="Confirm the error pattern is pool exhaustion rather than compute overload.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="worker"),
            rationale="Rule out the decoy worker deploy by reading worker logs directly.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "database_only_failure",
                    "affected_services": ["database", "api-gateway", "worker"],
                    "confidence": 0.8,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Localize the fault to the database config before remediating.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="database"),
            rationale="Roll back the offending database config rollout.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="restart_service", service="database"),
            rationale="Restart the database cleanly against the restored pool config.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Verify database pool health and write latency are back within SLO.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify gateway write-path traffic succeeds end-to-end.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def _gateway_auth_rollout_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="api-gateway"),
            rationale="Gateway is rejecting logins; read gateway logs to localize the rejection class.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="api-gateway"),
            rationale="Login rejection aligns with a recent auth middleware rollout; confirm deploy timing.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="worker"),
            rationale="Rule out the worker deploy explicitly rather than assuming.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "api_gateway_fault",
                    "affected_services": ["api-gateway", "worker"],
                    "confidence": 0.85,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Commit a calibrated hypothesis localizing to the gateway auth rollout.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"),
            rationale="Roll back the bad auth middleware rollout; no restart needed.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify that gateway login traffic now succeeds end-to-end.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Confirm the database is (and stayed) healthy throughout.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def _payment_webhook_misconfig_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="api-gateway"),
            rationale="Gateway owns the webhook handler; read logs for the signature-failure signature.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="api-gateway"),
            rationale="Recent gateway deploy is the most likely driver; confirm timing and version.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_metrics", service="database", metric="error_rate"),
            rationale="Confirm the database itself is not faulting — the webhook path writes to it but DB errors are steady.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "payment_webhook_regression",
                    "affected_services": ["api-gateway", "database"],
                    "confidence": 0.85,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Localize the fault to the gateway's webhook handler deploy before remediating.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"),
            rationale="Roll back the bad webhook handler deploy; no restart needed for this class of fault.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify a fresh Stripe webhook round-trip succeeds end-to-end.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Confirm subscriptions-table write path is fully restored.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def _schema_drift_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="api-gateway"),
            rationale="Gateway is reporting the PrismaClientKnownRequestError; inspect log text for the column name.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="api-gateway"),
            rationale="The error started with the latest gateway deploy; confirm which deploy and its scope.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="database"),
            rationale="Verify the database is healthy and the column genuinely isn't there — rules out DB-side corruption.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "schema_migration_mismatch",
                    "affected_services": ["api-gateway", "worker", "database"],
                    "confidence": 0.88,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Localize to the gateway deploy that shipped schema-expecting code without applying its migration.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="api-gateway"),
            rationale="Revert the gateway to a schema-compatible version; migration re-application is a separate workstream.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify /billing and /settings return 200 again.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Confirm DB read/write stayed healthy throughout.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


def _cache_stale_state_baseline() -> list[BaselineStep]:
    return [
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_logs", service="cache"),
            rationale="Support tickets point at stale/cross-user state; cache logs reveal the TTL change.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_deploys", service="cache"),
            rationale="Confirm the cache deploy that bumped TTL is the driver, not a gateway regression.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="query_metrics", service="database", metric="cpu"),
            rationale="Drop in DB read volume corroborates cache over-serving rather than a DB fault.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": "cache_ttl_regression",
                    "affected_services": ["cache", "api-gateway"],
                    "confidence": 0.85,
                    "recommended_next_action": "rollback_deploy",
                },
            ),
            rationale="Localize to the cache TTL deploy; gateway and DB are symptom-only.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="rollback_deploy", service="cache"),
            rationale="Roll back the TTL-bumping cache deploy.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="restart_service", service="cache"),
            rationale="Restart the cache to purge poisoned stale entries written under the bad TTL.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="end_to_end"),
            rationale="Verify fresh login shows own-user data, not stale snapshot.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="run_check", check_name="database_recovery"),
            rationale="Confirm DB read volume and cache-consistency window are back to baseline.",
        ),
        BaselineStep(
            action=UnifiedIncidentAction(action_type="declare_resolved"),
            rationale="Declare resolved only after objective checks pass.",
        ),
    ]


_BASELINE_BUILDERS = {
    "worker_deploy_cascade": _worker_cascade_baseline,
    "db_config_rollout": _db_config_rollout_baseline,
    "gateway_auth_rollout": _gateway_auth_rollout_baseline,
    "payment_webhook_misconfig": _payment_webhook_misconfig_baseline,
    "schema_drift_missing_migration": _schema_drift_baseline,
    "cache_stale_state": _cache_stale_state_baseline,
    # Round-2 Basic-tier templates (see basic_templates_extra.py).
    **extra_baselines(),
}


def _baseline_actions(scenario_id: str) -> list[BaselineStep]:
    template_id = SCENARIOS[scenario_id].get("template_id", scenario_id)
    builder = _BASELINE_BUILDERS.get(template_id)
    if builder is None:
        raise ValueError(f"No baseline for scenario_id {scenario_id!r}")
    return builder()


def list_baselines(scenario_id: str | None = None, include_procgen: bool = True) -> BaselineCatalog:
    if scenario_id is not None:
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario_id {scenario_id!r}")
        scenario_ids = [scenario_id]
    else:
        scenario_ids = [
            current_id
            for current_id, scenario in SCENARIOS.items()
            if include_procgen or not scenario.get("is_procgen", False)
        ]
    baselines = [
        BaselineDefinition(
            scenario_id=current_id,
            name="deterministic-remediation-baseline",
            description=SCENARIOS[current_id]["description"],
            optimal_ticks=SCENARIOS[current_id]["optimal_ticks"],
            actions=_baseline_actions(current_id),
        )
        for current_id in scenario_ids
    ]
    return BaselineCatalog(baselines=baselines)


def set_runtime_progress(progress: dict[str, Any]) -> None:
    global _RUNTIME_PROGRESS
    _RUNTIME_PROGRESS = deepcopy(progress)


def current_runtime_progress() -> dict[str, Any]:
    if _RUNTIME_PROGRESS is None:
        raise ValueError("Runtime progress is not initialized")
    return deepcopy(_RUNTIME_PROGRESS)


def grade_episode(state: dict[str, Any]):
    from .grader import UnifiedIncidentGrader

    scenario_id = state.get("scenario_id", DEFAULT_SCENARIO_ID)
    return UnifiedIncidentGrader().build_report(state, get_scenario(scenario_id))
