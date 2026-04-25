"""Run the scripted-optimal baseline across all 12 templates × 5 procgen variants
and print a summary table. This is the smoke-check that the env is healthy and
the baseline ceiling is preserved.

Usage:
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --templates-only
    python scripts/eval_baseline.py --episodes-per-scenario 3
    python scripts/eval_baseline.py --output eval/results/baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.challenge import (
    SCENARIOS,
    list_baselines,
)
from unified_incident_env.server.environment import UnifiedIncidentEnvironment


def run_one(scenario_id: str) -> dict:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    baseline = list_baselines(scenario_id=scenario_id).baselines[0]
    for step in baseline.actions:
        obs = env.step(step.action)
        if obs.done:
            break
    return {
        "scenario_id": scenario_id,
        "template_id": SCENARIOS[scenario_id].get("template_id", scenario_id),
        "is_procgen": SCENARIOS[scenario_id].get("is_procgen", False),
        "final_score": float(obs.final_score),
        "incident_resolved": bool(obs.incident_resolved),
        "tick_count": int(obs.tick_count),
        "breakdown": dict(obs.score_breakdown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates-only", action="store_true",
                        help="Run only the 12 base templates, skip procgen variants.")
    parser.add_argument("--episodes-per-scenario", type=int, default=1,
                        help="Number of times to run each scenario (deterministic, so default 1).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSONL output path.")
    args = parser.parse_args()

    if args.templates_only:
        scenario_ids = sorted(sid for sid, sc in SCENARIOS.items() if not sc.get("is_procgen"))
    else:
        scenario_ids = sorted(SCENARIOS.keys())

    results = []
    for sid in scenario_ids:
        for _ in range(args.episodes_per_scenario):
            r = run_one(sid)
            results.append(r)

    print(f"\n{'scenario':<40} {'score':>7}  {'resolved':>9}  {'ticks':>5}")
    print("-" * 70)
    for r in results:
        flag = "OK" if r["incident_resolved"] else "X"
        print(f"{r['scenario_id']:<40} {r['final_score']:>7.3f}  {flag:>9}  {r['tick_count']:>5}")

    print()
    by_template: dict[str, list[float]] = {}
    for r in results:
        by_template.setdefault(r["template_id"], []).append(r["final_score"])
    print(f"{'template':<40} {'mean':>7}  {'min':>7}  {'max':>7}  {'n':>3}")
    print("-" * 70)
    for tid, scores in sorted(by_template.items()):
        print(f"{tid:<40} {mean(scores):>7.3f}  {min(scores):>7.3f}  {max(scores):>7.3f}  {len(scores):>3}")

    overall_mean = mean(r["final_score"] for r in results)
    overall_resolved = sum(r["incident_resolved"] for r in results)
    print(f"\nOverall: mean={overall_mean:.3f}, resolved={overall_resolved}/{len(results)}")
    if overall_mean > 0.80:
        print("WARNING: scripted baseline ceiling exceeded 0.80 — see docs/REWARD_DESIGN.md §4")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nWrote {len(results)} rows -> {out}")


if __name__ == "__main__":
    main()
