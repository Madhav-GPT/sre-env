"""Batch evaluation for one or more models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..server.challenge import SCENARIOS
from .action_adapter import LenientActionAdapter, StrictActionParser
from .backend import OpenAICompatibleBackend
from .run_episode import EpisodeRunner
from .trajectory_store import TrajectoryStore
from .types import EvalScenarioResult, EvalSummary


def summarize(results: list[EvalScenarioResult], mode: str) -> EvalSummary:
    success_rate = (
        sum(1 for result in results if result.success) / len(results) if results else 0.0
    )
    avg_score = (
        sum(result.final_score for result in results) / len(results) if results else 0.0
    )
    schema_failure_rate = (
        sum(1 for result in results if result.schema_failure) / len(results)
        if results
        else 0.0
    )

    by_model: dict[str, dict[str, float]] = {}
    by_scenario: dict[str, dict[str, float]] = {}
    for result in results:
        model_bucket = by_model.setdefault(
            result.model_name,
            {"runs": 0.0, "successes": 0.0, "score_sum": 0.0, "schema_failures": 0.0},
        )
        model_bucket["runs"] += 1
        model_bucket["successes"] += 1.0 if result.success else 0.0
        model_bucket["score_sum"] += result.final_score
        model_bucket["schema_failures"] += 1.0 if result.schema_failure else 0.0

        scenario_bucket = by_scenario.setdefault(
            result.scenario_id,
            {"runs": 0.0, "successes": 0.0, "score_sum": 0.0},
        )
        scenario_bucket["runs"] += 1
        scenario_bucket["successes"] += 1.0 if result.success else 0.0
        scenario_bucket["score_sum"] += result.final_score

    for bucket in by_model.values():
        runs = bucket["runs"] or 1.0
        bucket["success_rate"] = round(bucket["successes"] / runs, 4)
        bucket["avg_score"] = round(bucket["score_sum"] / runs, 4)
        bucket["schema_failure_rate"] = round(bucket["schema_failures"] / runs, 4)
        del bucket["score_sum"]
        del bucket["successes"]
        del bucket["schema_failures"]

    for bucket in by_scenario.values():
        runs = bucket["runs"] or 1.0
        bucket["success_rate"] = round(bucket["successes"] / runs, 4)
        bucket["avg_score"] = round(bucket["score_sum"] / runs, 4)
        del bucket["score_sum"]
        del bucket["successes"]

    return EvalSummary(
        mode=mode,
        results=results,
        success_rate=round(success_rate, 4),
        avg_score=round(avg_score, 4),
        schema_failure_rate=round(schema_failure_rate, 4),
        by_model=by_model,
        by_scenario=by_scenario,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("API_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or "local",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--episodes-output",
        default="outputs/trainer/episodes.jsonl",
    )
    args = parser.parse_args()

    backend = OpenAICompatibleBackend(
        base_url=args.api_base_url,
        api_key=args.api_key,
    )
    parser_impl = StrictActionParser() if args.mode == "strict" else LenientActionAdapter()
    episode_store = TrajectoryStore(Path(args.episodes_output))

    results: list[EvalScenarioResult] = []
    for model_name in args.models:
        runner = EpisodeRunner(
            backend=backend,
            parser=parser_impl,
            model_name=model_name,
            base_url=args.base_url,
        )
        for scenario_id in SCENARIOS:
            episode = runner.run(scenario_id=scenario_id, mode=args.mode)
            episode_store.append_episode(episode)
            results.append(
                EvalScenarioResult(
                    model_name=model_name,
                    scenario_id=scenario_id,
                    success=episode.success,
                    final_score=episode.final_score,
                    failure_reason=episode.failure_reason,
                    schema_failure=bool(
                        episode.failure_reason
                        and episode.failure_reason.startswith("parse_failure")
                    ),
                    elapsed_s=episode.elapsed_s,
                )
            )

    summary = summarize(results, mode=args.mode)
    output_path = Path(
        args.output
        or f"outputs/trainer/{args.mode}_eval_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
