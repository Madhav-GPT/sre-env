"""Deterministic scripted baseline for the final preset pack."""

from __future__ import annotations

import argparse
import json

from ..client import UnifiedIncidentEnv
from ..models import PostmortemPayload, UnifiedIncidentAction
from ..server.challenge import BASELINES, SCENARIOS


POSTMORTEMS = {
    "database_sqli_outage": PostmortemPayload(
        root_cause="SQL injection in login exhausted and crashed the database.",
        attack_vector="Unsanitized SQL input triggered abusive database load.",
        timeline=[
            "Queried database logs",
            "Inspected code",
            "Patched SQL injection",
            "Verified exploit blocked",
            "Restarted database",
        ],
        remediation_steps=["Use parameterized query", "Restart database"],
        prevention_steps=["Parameterized queries", "DB abuse alerting"],
    ),
    "cache_abuse_broken_access_control": PostmortemPayload(
        root_cause="Broken access control on the internal admin endpoint caused a cache and database cascade.",
        attack_vector="Missing authorization let attackers abuse the internal admin path.",
        timeline=[
            "Queried cache metrics",
            "Queried api-gateway dependencies",
            "Inspected code",
            "Enforced admin role",
            "Verified exploit blocked",
            "Restarted cache",
            "Restarted database",
        ],
        remediation_steps=["Enforce admin role", "Restart cache", "Restart database"],
        prevention_steps=["Authorization checks", "Admin role enforcement", "Rate limits"],
    ),
    "worker_bad_deploy_command_injection": PostmortemPayload(
        root_cause="A bad worker deploy plus command injection repeatedly poisoned downstream services.",
        attack_vector="Shell-based worker commands accepted unsafe filenames and kept replaying corruption.",
        timeline=[
            "Queried worker logs",
            "Inspected code",
            "Patched command injection",
            "Verified exploit blocked",
            "Rolled back worker",
            "Restarted database",
        ],
        remediation_steps=["Avoid shell", "Rollback worker", "Restart database"],
        prevention_steps=["Avoid shell", "Input validation", "Safer deploy checks"],
    ),
}


def plan_for_scenario(scenario_id: str) -> list[UnifiedIncidentAction]:
    steps = [step.action for step in BASELINES[scenario_id].actions]
    steps.append(
        UnifiedIncidentAction(
            action_type="submit_postmortem",
            postmortem=POSTMORTEMS[scenario_id],
        )
    )
    return steps


def run_scenario(base_url: str, scenario_id: str) -> dict[str, object]:
    with UnifiedIncidentEnv(base_url=base_url).sync() as env:
        env.reset(scenario_id=scenario_id)
        final = None
        for action in plan_for_scenario(scenario_id):
            final = env.step(action).observation
        assert final is not None
        return {
            "scenario_id": scenario_id,
            "success": bool(final.done and final.incident_resolved),
            "final_score": final.final_score,
            "workflow_stage": final.workflow_stage,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=UnifiedIncidentEnv.DEFAULT_BASE_URL)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    args = parser.parse_args()

    scenario_ids = [args.scenario] if args.scenario else list(SCENARIOS)
    results = [run_scenario(args.base_url, scenario_id) for scenario_id in scenario_ids]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
