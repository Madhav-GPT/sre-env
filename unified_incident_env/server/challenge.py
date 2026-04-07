"""Final preset scenario catalog, baselines, and runtime helpers."""

from __future__ import annotations

from copy import deepcopy

from ..models import (
    BaselineCatalog,
    BaselineDefinition,
    BaselineStep,
    ScenarioCatalog,
    ScenarioSummary,
    UnifiedIncidentAction,
)

DEFAULT_SCENARIO_ID = "database_sqli_outage"

SCENARIOS: dict[str, dict] = {
    "database_sqli_outage": {
        "id": "database_sqli_outage",
        "difficulty": "easy",
        "name": "Database SQLi Outage",
        "description": (
            "Login SQL injection abuse overloads and crashes the database, which then drives gateway 502s."
        ),
        "common_trap": "Restarting api-gateway only treats the symptom; the database and exploit path are still the root issue.",
        "max_ticks": 8,
        "optimal_ticks": 8,
        "root_cause": "SQL injection in the login endpoint crashed the database.",
        "attack_vector": "Unsanitized SQL input triggered abusive database load.",
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 71.0,
                "memory_pct": 49.0,
                "error_rate_pct": 24.0,
                "latency_ms": 710.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 24.0,
                "memory_pct": 29.0,
                "error_rate_pct": 0.0,
                "latency_ms": 18.0,
            },
            "database": {
                "status": "crashed",
                "cpu_pct": 99.0,
                "memory_pct": 98.0,
                "error_rate_pct": 100.0,
                "latency_ms": 0.0,
            },
            "worker": {
                "status": "healthy",
                "cpu_pct": 20.0,
                "memory_pct": 27.0,
                "error_rate_pct": 0.0,
                "latency_ms": 12.0,
            },
        },
        "initial_alerts": [
            {
                "service": "api-gateway",
                "severity": "critical",
                "message": "High 502 rate on login and checkout.",
            },
            {
                "service": "database",
                "severity": "critical",
                "message": "Database process not responding.",
            },
        ],
        "logs": {
            "api-gateway": "Gateway retries are failing because the database is down; 502s spiking across auth traffic.",
            "cache": "Cache is healthy; upstream database lookups are failing.",
            "database": (
                "SECURITY_ALERT root cause confirmed: login query used raw string concatenation; "
                "payload `' OR 1=1 --` triggered abusive query load, exhausted the connection pool, "
                "and crashed the database process."
            ),
            "worker": "Worker is healthy and unrelated to this outage.",
        },
        "metrics": {
            "database": {
                "cpu": "Database CPU was pinned at 99% before the crash.",
                "memory": "Database memory hit 98% immediately before process exit.",
            },
            "api-gateway": {
                "error_rate": "Gateway error rate is 24%, almost all 502s.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> cache -> database",
            "database": "database serves login and order state",
        },
        "evidence_rules": [
            {
                "id": "db_logs_root",
                "action_type": "query_logs",
                "service": "database",
                "detail": "Database logs explicitly confirm SQL injection caused the outage.",
                "identifies_root_cause": True,
                "unlocks_security": True,
                "unlock_reason": "Database root-cause logs exposed the vulnerable login path, so the security subquest is now active.",
            },
            {
                "id": "db_metrics_memory",
                "action_type": "query_metrics",
                "service": "database",
                "metric": "memory",
                "detail": "Database memory spike confirmed crash pressure from abusive login traffic.",
                "identifies_root_cause": False,
                "unlocks_security": False,
            },
            {
                "id": "gateway_dependencies",
                "action_type": "query_dependencies",
                "service": "api-gateway",
                "detail": "Gateway depends on the database path.",
                "identifies_root_cause": False,
                "unlocks_security": False,
            },
        ],
        "unlock_threshold": 1,
        "security": {
            "hint": "The login handler must bind user input instead of concatenating SQL.",
            "code_context": (
                "def login(username, password):\n"
                "    query = \"SELECT id FROM users WHERE username='\" + username + "
                "\"' AND password='\" + password + \"'\"\n"
                "    return db.execute(query).fetchone()\n"
            ),
            "correct_vulnerability": "sql_injection",
            "correct_patch": "parameterized_query",
            "patch_options": [
                {"id": "parameterized_query", "label": "Use a parameterized query"},
                {"id": "strip_quotes", "label": "Strip quotes from input"},
                {"id": "disable_login", "label": "Disable the login endpoint"},
            ],
            "verify_outcomes": {
                "parameterized_query": {
                    "exploit_blocked": True,
                    "functionality_preserved": True,
                    "message": "Exploit blocked and legitimate logins still work.",
                },
                "strip_quotes": {
                    "exploit_blocked": False,
                    "functionality_preserved": True,
                    "message": "Quote stripping is bypassable; exploit still works.",
                },
                "disable_login": {
                    "exploit_blocked": True,
                    "functionality_preserved": False,
                    "message": "Exploit blocked, but legitimate users are locked out.",
                },
            },
        },
        "recovery_sequence": [
            {
                "action_type": "restart_service",
                "service": "database",
                "requires_security_completion": True,
                "message": "Database restarted cleanly after the SQL injection path was closed.",
                "updates": {
                    "database": {
                        "status": "healthy",
                        "cpu_pct": 37.0,
                        "memory_pct": 43.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 19.0,
                    },
                    "api-gateway": {
                        "status": "healthy",
                        "cpu_pct": 33.0,
                        "memory_pct": 34.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 41.0,
                    },
                },
            }
        ],
        "trap_actions": [
            {
                "action_type": "restart_service",
                "service": "api-gateway",
                "message": "Restarting api-gateway first does not help while the database is still crashed.",
            },
            {
                "action_type": "restart_service",
                "service": "cache",
                "message": "Restarting cache is a trap; cache is not the failing root cause.",
            },
        ],
        "postmortem_keywords": {
            "root_cause": ["sql injection", "database"],
            "attack_vector": ["unsanitized sql", "login", "input"],
            "remediation": ["parameterized query", "restart database"],
            "prevention": ["parameterized queries", "db abuse alerting", "alerting"],
        },
    },
    "cache_abuse_broken_access_control": {
        "id": "cache_abuse_broken_access_control",
        "difficulty": "medium",
        "name": "Cache Abuse Broken Access Control",
        "description": (
            "An internal admin endpoint lacks proper authorization and abuse cascades through cache and database."
        ),
        "common_trap": "Restarting database or api-gateway first does not close the abused admin path.",
        "max_ticks": 20,
        "optimal_ticks": 10,
        "root_cause": "Broken access control exposed an internal admin endpoint and caused a cache/database cascade.",
        "attack_vector": "Missing admin authorization let attackers abuse an internal cache-management route.",
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 63.0,
                "memory_pct": 44.0,
                "error_rate_pct": 11.0,
                "latency_ms": 420.0,
            },
            "cache": {
                "status": "crashed",
                "cpu_pct": 95.0,
                "memory_pct": 93.0,
                "error_rate_pct": 100.0,
                "latency_ms": 0.0,
            },
            "database": {
                "status": "degraded",
                "cpu_pct": 88.0,
                "memory_pct": 74.0,
                "error_rate_pct": 16.0,
                "latency_ms": 240.0,
            },
            "worker": {
                "status": "healthy",
                "cpu_pct": 19.0,
                "memory_pct": 25.0,
                "error_rate_pct": 0.0,
                "latency_ms": 11.0,
            },
        },
        "initial_alerts": [
            {
                "service": "cache",
                "severity": "critical",
                "message": "Cache repeated crash and overload cycle.",
            },
            {
                "service": "database",
                "severity": "critical",
                "message": "Database sustained high CPU after cache failures.",
            },
            {
                "service": "api-gateway",
                "severity": "warning",
                "message": "Gateway latency spiking across cache-backed routes.",
            },
        ],
        "logs": {
            "api-gateway": (
                "Repeated calls to /internal/admin/cache-purge are causing downstream churn and latency spikes."
            ),
            "cache": (
                "SECURITY_ALERT unauthorized requests hit the internal admin purge endpoint; "
                "rebuild storm repeatedly crashed cache."
            ),
            "database": "Database is serving cache-miss flood traffic and remains degraded.",
            "worker": "Worker is healthy and not on the critical path.",
        },
        "metrics": {
            "cache": {
                "cpu": "Cache CPU remains pinned during repeated rebuild cycles.",
                "memory": "Cache memory spiked before each crash.",
            },
            "database": {
                "cpu": "Database CPU is elevated due to cache misses.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> cache -> database",
            "cache": "cache is flushed by internal admin handlers behind api-gateway",
        },
        "evidence_rules": [
            {
                "id": "cache_metrics_root",
                "action_type": "query_metrics",
                "service": "cache",
                "metric": "cpu",
                "detail": "Cache metrics showed a rebuild storm.",
                "identifies_root_cause": True,
                "unlocks_security": False,
            },
            {
                "id": "gateway_dependencies_unlock",
                "action_type": "query_dependencies",
                "service": "api-gateway",
                "detail": "Dependency analysis pointed to an internal admin chain through api-gateway.",
                "identifies_root_cause": False,
                "unlocks_security": True,
                "unlock_reason": "The dependency chain exposed an internal admin route, so code inspection is now justified.",
            },
            {
                "id": "cache_logs_alt_unlock",
                "action_type": "query_logs",
                "service": "cache",
                "detail": "Cache logs exposed the missing admin authorization issue.",
                "identifies_root_cause": True,
                "unlocks_security": True,
                "unlock_reason": "Cache logs exposed the authorization flaw directly, so the security subquest is now active.",
            },
        ],
        "unlock_threshold": 2,
        "security": {
            "hint": "The internal admin endpoint must enforce proper server-side authorization.",
            "code_context": (
                "def purge_cache(request, current_user):\n"
                "    if not current_user:\n"
                "        raise UnauthorizedError()\n"
                "    return cache.purge(request.query_params['key'])\n"
            ),
            "correct_vulnerability": "broken_access_control",
            "correct_patch": "enforce_admin_role",
            "patch_options": [
                {"id": "enforce_admin_role", "label": "Enforce admin role server-side"},
                {"id": "hide_admin_link", "label": "Hide the admin link in the UI"},
                {"id": "deny_all_traffic", "label": "Block all traffic to the endpoint"},
            ],
            "verify_outcomes": {
                "enforce_admin_role": {
                    "exploit_blocked": True,
                    "functionality_preserved": True,
                    "message": "Abuse path closed and valid admins still work.",
                },
                "hide_admin_link": {
                    "exploit_blocked": False,
                    "functionality_preserved": True,
                    "message": "UI-only hiding fails; direct requests still work.",
                },
                "deny_all_traffic": {
                    "exploit_blocked": True,
                    "functionality_preserved": False,
                    "message": "Abuse closed, but legitimate admin usage is broken.",
                },
            },
        },
        "recovery_sequence": [
            {
                "action_type": "restart_service",
                "service": "cache",
                "requires_security_completion": True,
                "message": "Cache recovered and stayed healthy once the abuse path was closed.",
                "updates": {
                    "cache": {
                        "status": "healthy",
                        "cpu_pct": 36.0,
                        "memory_pct": 42.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 12.0,
                    }
                },
            },
            {
                "action_type": "restart_service",
                "service": "database",
                "requires_security_completion": True,
                "message": "Database stabilized after cache recovered and load normalized.",
                "updates": {
                    "database": {
                        "status": "healthy",
                        "cpu_pct": 35.0,
                        "memory_pct": 41.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 23.0,
                    },
                    "api-gateway": {
                        "status": "healthy",
                        "cpu_pct": 31.0,
                        "memory_pct": 33.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 39.0,
                    },
                },
            },
        ],
        "trap_actions": [
            {
                "action_type": "restart_service",
                "service": "database",
                "message": "Restarting database first is a trap; the abuse path will overload it again.",
            },
            {
                "action_type": "restart_service",
                "service": "api-gateway",
                "message": "Restarting api-gateway first treats a symptom, not the root cause.",
            },
        ],
        "postmortem_keywords": {
            "root_cause": ["broken access control", "cache", "admin endpoint"],
            "attack_vector": ["authorization", "admin", "internal"],
            "remediation": ["enforce admin role", "restart cache", "restart database"],
            "prevention": ["authorization", "admin role", "rate limits"],
        },
    },
    "worker_bad_deploy_command_injection": {
        "id": "worker_bad_deploy_command_injection",
        "difficulty": "hard",
        "name": "Worker Bad Deploy Command Injection",
        "description": (
            "A bad worker deployment plus command injection repeatedly poisons downstream services."
        ),
        "common_trap": "Restarting downstream services first does not help while the worker remains unsafe.",
        "max_ticks": 20,
        "optimal_ticks": 9,
        "root_cause": "A bad worker deploy plus command injection repeatedly poisoned downstream systems.",
        "attack_vector": "Shell command construction in the worker allowed repeated command injection.",
        "initial_services": {
            "api-gateway": {
                "status": "degraded",
                "cpu_pct": 58.0,
                "memory_pct": 41.0,
                "error_rate_pct": 9.0,
                "latency_ms": 370.0,
            },
            "cache": {
                "status": "healthy",
                "cpu_pct": 23.0,
                "memory_pct": 28.0,
                "error_rate_pct": 0.0,
                "latency_ms": 15.0,
            },
            "database": {
                "status": "degraded",
                "cpu_pct": 79.0,
                "memory_pct": 66.0,
                "error_rate_pct": 13.0,
                "latency_ms": 210.0,
            },
            "worker": {
                "status": "degraded",
                "cpu_pct": 94.0,
                "memory_pct": 87.0,
                "error_rate_pct": 31.0,
                "latency_ms": 160.0,
            },
        },
        "initial_alerts": [
            {
                "service": "worker",
                "severity": "critical",
                "message": "Worker abnormal memory growth and process churn.",
            },
            {
                "service": "database",
                "severity": "critical",
                "message": "Database connection instability from worker retries.",
            },
            {
                "service": "api-gateway",
                "severity": "warning",
                "message": "Partial outage across export-backed routes.",
            },
        ],
        "logs": {
            "api-gateway": "Gateway requests are stalling on worker-backed exports and reading stale downstream state.",
            "cache": "Cache is healthy; no primary fault present here.",
            "database": "Database connections are unstable because worker retries keep replaying corrupt jobs.",
            "worker": (
                "SECURITY_ALERT export worker shells out with unsanitized user-controlled filenames; "
                "current deploy removed safe command wrapper and corruption repeats."
            ),
        },
        "metrics": {
            "database": {
                "cpu": "Database CPU remains elevated from repeated worker retry churn.",
                "latency": "Database latency spikes when corrupt export jobs replay.",
            },
            "worker": {
                "memory": "Worker memory and process churn indicate a bad deploy plus repeated exploit path.",
            },
        },
        "dependencies": {
            "api-gateway": "api-gateway -> worker -> database",
            "worker": "worker writes export state into database and feeds api-gateway completion paths",
        },
        "evidence_rules": [
            {
                "id": "worker_logs_root",
                "action_type": "query_logs",
                "service": "worker",
                "detail": "Worker logs exposed the command injection clue and bad deploy link.",
                "identifies_root_cause": True,
                "unlocks_security": True,
                "unlock_reason": "Worker logs already expose the exploit path, so the security subquest is now active.",
            },
            {
                "id": "db_metrics_support",
                "action_type": "query_metrics",
                "service": "database",
                "metric": "cpu",
                "detail": "Database metrics confirmed downstream instability from worker retries.",
                "identifies_root_cause": False,
                "unlocks_security": False,
            },
            {
                "id": "gateway_dependency_chain",
                "action_type": "query_dependencies",
                "service": "api-gateway",
                "detail": "Dependency analysis exposed the worker-to-database-to-gateway chain.",
                "identifies_root_cause": False,
                "unlocks_security": True,
                "unlock_reason": "The dependency chain points back to the worker exploit path, so the security subquest is now active.",
            },
        ],
        "unlock_threshold": 1,
        "security": {
            "hint": "The worker must stop passing filenames into a shell command.",
            "code_context": (
                "def build_export(filename):\n"
                "    cmd = '/usr/bin/zip /tmp/out.zip ' + filename\n"
                "    return subprocess.check_output(cmd, shell=True)\n"
            ),
            "correct_vulnerability": "command_injection",
            "correct_patch": "avoid_shell",
            "patch_options": [
                {"id": "avoid_shell", "label": "Use argv and avoid shell invocation"},
                {"id": "sanitize_quotes_only", "label": "Sanitize quotes only"},
                {"id": "disable_worker_commands", "label": "Disable worker commands"},
            ],
            "verify_outcomes": {
                "avoid_shell": {
                    "exploit_blocked": True,
                    "functionality_preserved": True,
                    "message": "Exploit blocked and worker exports still function.",
                },
                "sanitize_quotes_only": {
                    "exploit_blocked": False,
                    "functionality_preserved": True,
                    "message": "Quote-only sanitization is bypassable; exploit path remains.",
                },
                "disable_worker_commands": {
                    "exploit_blocked": True,
                    "functionality_preserved": False,
                    "message": "Exploit blocked, but core worker functionality is disabled.",
                },
            },
        },
        "recovery_sequence": [
            {
                "action_type": "rollback_deploy",
                "service": "worker",
                "requires_security_completion": True,
                "message": "Worker rolled back to the safe version after the exploit path was closed.",
                "updates": {
                    "worker": {
                        "status": "healthy",
                        "cpu_pct": 37.0,
                        "memory_pct": 40.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 26.0,
                    }
                },
            },
            {
                "action_type": "restart_service",
                "service": "database",
                "requires_security_completion": True,
                "message": "Database restarted cleanly after worker poison traffic stopped, and api-gateway stabilized.",
                "updates": {
                    "database": {
                        "status": "healthy",
                        "cpu_pct": 34.0,
                        "memory_pct": 38.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 24.0,
                    },
                    "api-gateway": {
                        "status": "healthy",
                        "cpu_pct": 29.0,
                        "memory_pct": 31.0,
                        "error_rate_pct": 0.0,
                        "latency_ms": 36.0,
                    }
                },
            },
        ],
        "trap_actions": [
            {
                "action_type": "restart_service",
                "service": "database",
                "message": "Restarting database first is a trap; the bad worker path will re-poison it.",
            },
            {
                "action_type": "restart_service",
                "service": "api-gateway",
                "message": "Restarting api-gateway first treats a symptom while the worker remains unsafe.",
            },
            {
                "action_type": "rollback_deploy",
                "service": "worker",
                "message": "Rolling back before closing the command injection path is a trap; the exploit still persists.",
                "requires_security_incomplete": True,
            },
        ],
        "postmortem_keywords": {
            "root_cause": ["bad deploy", "worker", "command injection"],
            "attack_vector": ["shell", "filename", "command injection"],
            "remediation": ["avoid shell", "rollback", "restart database", "restart api-gateway"],
            "prevention": ["avoid shell", "input validation", "deploy"],
        },
    },
}

BASELINES: dict[str, BaselineDefinition] = {
    "database_sqli_outage": BaselineDefinition(
        scenario_id="database_sqli_outage",
        name="Easy baseline",
        description="Deterministic happy path for the easy SQL injection outage.",
        optimal_ticks=8,
        actions=[
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="query_logs",
                    service="database",
                ),
                rationale="Find the SQL injection clue in the root-cause logs.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="inspect_code"),
                rationale="Reveal the vulnerable login code.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="classify_vulnerability",
                    vulnerability_type="sql_injection",
                ),
                rationale="Classify the vulnerability correctly.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="apply_patch",
                    patch_id="parameterized_query",
                ),
                rationale="Apply the correct patch.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="verify_security_fix"),
                rationale="Verify exploit blocking and functionality.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="submit_security_fix"),
                rationale="Complete the security subquest.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="restart_service",
                    service="database",
                ),
                rationale="Recover the failed database after the exploit is closed.",
            ),
        ],
    ),
    "cache_abuse_broken_access_control": BaselineDefinition(
        scenario_id="cache_abuse_broken_access_control",
        name="Medium baseline",
        description="Deterministic happy path for the broken-access-control cascade.",
        optimal_ticks=10,
        actions=[
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="query_metrics",
                    service="cache",
                    metric="cpu",
                ),
                rationale="Confirm the cache rebuild storm.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="query_dependencies",
                    service="api-gateway",
                ),
                rationale="Expose the suspicious internal admin chain.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="inspect_code"),
                rationale="Reveal the vulnerable admin endpoint.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="classify_vulnerability",
                    vulnerability_type="broken_access_control",
                ),
                rationale="Classify the authz flaw.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="apply_patch",
                    patch_id="enforce_admin_role",
                ),
                rationale="Apply the correct access-control patch.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="verify_security_fix"),
                rationale="Verify the abuse path is closed.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="submit_security_fix"),
                rationale="Complete the security subquest.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="restart_service",
                    service="cache",
                ),
                rationale="Recover cache first.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="restart_service",
                    service="database",
                ),
                rationale="Recover the database after cache stabilizes.",
            ),
        ],
    ),
    "worker_bad_deploy_command_injection": BaselineDefinition(
        scenario_id="worker_bad_deploy_command_injection",
        name="Hard baseline",
        description="Deterministic happy path for the bad-deploy command-injection incident.",
        optimal_ticks=9,
        actions=[
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="query_logs",
                    service="worker",
                ),
                rationale="Find the worker exploit clue.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="inspect_code"),
                rationale="Reveal the vulnerable worker code.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="classify_vulnerability",
                    vulnerability_type="command_injection",
                ),
                rationale="Classify the vulnerability correctly.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="apply_patch",
                    patch_id="avoid_shell",
                ),
                rationale="Apply the correct worker patch.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="verify_security_fix"),
                rationale="Verify exploit blocking and function retention.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(action_type="submit_security_fix"),
                rationale="Complete the security subquest.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="rollback_deploy",
                    service="worker",
                ),
                rationale="Rollback the bad deploy after closing the exploit path.",
            ),
            BaselineStep(
                action=UnifiedIncidentAction(
                    action_type="restart_service",
                    service="database",
                ),
                rationale="Recover the poisoned database state.",
            ),
        ],
    ),
}

_CURRENT_PROGRESS: dict = {
    "episode_id": "bootstrap",
    "step_count": 0,
    "scenario_id": DEFAULT_SCENARIO_ID,
    "difficulty": "easy",
    "current_tick": 0,
    "max_ticks": SCENARIOS[DEFAULT_SCENARIO_ID]["max_ticks"],
    "workflow_stage": "diagnosis",
    "active_alerts": [],
    "service_health": {},
    "discovered_evidence": [],
    "identified_root_cause": None,
    "security_subquest_status": "locked",
    "security_context": {
        "code_visible": False,
        "selected_vulnerability": None,
        "selected_patch": None,
        "exploit_blocked": None,
        "functionality_preserved": None,
    },
    "security_fix_submitted": False,
    "incident_resolved": False,
    "postmortem_submitted": False,
    "cumulative_reward": 0.0,
    "cumulative_score": 0.0,
    "score_breakdown": {
        "infrastructure_score": 0.0,
        "security_score": 0.0,
        "efficiency_score": 0.10,
        "postmortem_score": 0.0,
    },
    "wasteful_ticks": 0,
    "last_action_result": "",
}


def get_scenario(scenario_id: str) -> dict:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario_id {scenario_id!r}. Valid: {valid}") from exc


def scenario_for_difficulty(difficulty: str) -> dict:
    for scenario in SCENARIOS.values():
        if scenario["difficulty"] == difficulty:
            return scenario
    valid = ", ".join(sorted({scenario["difficulty"] for scenario in SCENARIOS.values()}))
    raise ValueError(f"Unknown difficulty {difficulty!r}. Valid: {valid}")


def list_scenarios(difficulty: str | None = None) -> ScenarioCatalog:
    difficulties = sorted(
        {scenario["difficulty"] for scenario in SCENARIOS.values()}
    )
    if difficulty is not None and difficulty not in difficulties:
        valid = ", ".join(difficulties)
        raise ValueError(f"Unknown difficulty {difficulty!r}. Valid: {valid}")
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
        if difficulty is None or scenario["difficulty"] == difficulty
    ]
    return ScenarioCatalog(
        default_scenario_id=DEFAULT_SCENARIO_ID,
        available_difficulties=difficulties,
        filtered_difficulty=difficulty,
        scenarios=scenarios,
    )


def list_baselines(scenario_id: str | None = None) -> BaselineCatalog:
    if scenario_id is None:
        return BaselineCatalog(baselines=list(BASELINES.values()))
    get_scenario(scenario_id)
    return BaselineCatalog(baselines=[BASELINES[scenario_id]])


def set_runtime_progress(state: dict) -> None:
    global _CURRENT_PROGRESS
    _CURRENT_PROGRESS = deepcopy(state)


def current_runtime_progress() -> dict:
    return deepcopy(_CURRENT_PROGRESS)


def grade_episode(state: dict):
    from .grader import UnifiedIncidentGrader

    scenario_id = state.get("scenario_id", DEFAULT_SCENARIO_ID)
    return UnifiedIncidentGrader().build_report(state, get_scenario(scenario_id))
