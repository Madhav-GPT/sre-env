"""Deterministic scripted baseline for the honest narrow incident environment."""

from __future__ import annotations

import argparse
import json

from ..client import UnifiedIncidentEnv
from ..server.challenge import DEFAULT_SCENARIO_ID, SCENARIOS, list_baselines


def plan_for_scenario(scenario_id: str):
    catalog = list_baselines(scenario_id)
    return [step.action for step in catalog.baselines[0].actions]


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
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO_ID)
    args = parser.parse_args()

    results = [run_scenario(args.base_url, args.scenario)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
