"""Simple walkthrough that prints a full episode interaction."""

from __future__ import annotations

import argparse
import json

from ..client import UnifiedIncidentEnv
from .baseline_agent import plan_for_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=UnifiedIncidentEnv.DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--scenario",
        default="easy_sqli_db_outage",
    )
    args = parser.parse_args()

    with UnifiedIncidentEnv(base_url=args.base_url).sync() as env:
        reset = env.reset(scenario_id=args.scenario).observation
        print(json.dumps({"reset": reset.model_dump()}, indent=2))
        for action in plan_for_scenario(args.scenario):
            step = env.step(action).observation
            print(
                json.dumps(
                    {
                        "action": action.model_dump(exclude_none=True),
                        "observation": step.model_dump(),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
