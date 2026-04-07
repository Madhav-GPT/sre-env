"""Episode runner for strict and lenient training/eval modes."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from ..client import UnifiedIncidentEnv
from ..models import UnifiedIncidentAction
from ..scripts.baseline_agent import plan_for_scenario
from ..server.challenge import SCENARIOS
from ..server.environment import UnifiedIncidentEnvironment
from .action_adapter import LenientActionAdapter, StrictActionParser
from .backend import OpenAICompatibleBackend
from .prompts import build_repair_request, build_runtime_request
from .trajectory_memory import CorrectionMemory
from .trajectory_store import TrajectoryStore
from .types import EpisodeRecord, ModelRequest, StepRecord


class EpisodeRunner:
    """Runs one model through one scenario and records a full trajectory."""

    def __init__(
        self,
        *,
        backend,
        parser,
        model_name: str,
        base_url: str | None = None,
        env_factory=None,
        correction_memory: CorrectionMemory | None = None,
    ) -> None:
        self.backend = backend
        self.parser = parser
        self.model_name = model_name
        self.base_url = base_url
        self.env_factory = env_factory
        self.correction_memory = correction_memory or CorrectionMemory()

    def run(
        self,
        scenario_id: str,
        mode: str,
        *,
        episode_id: int | None = None,
        model_version: str | None = None,
    ) -> EpisodeRecord:
        started = time.perf_counter()
        scenario = SCENARIOS[scenario_id]
        teacher_plan = plan_for_scenario(scenario_id)
        step_records: list[StepRecord] = []
        failure_reason: str | None = None
        final_score = 0.0
        success = False
        steps_taken = 0
        previous_action: dict[str, Any] | None = None
        repeated_no_progress = 0

        with self._env_context() as env:
            observation = self._unwrap_observation(env.reset(scenario_id=scenario_id))
            while not observation.done:
                teacher_action = (
                    teacher_plan[steps_taken].model_dump(exclude_none=True)
                    if steps_taken < len(teacher_plan)
                    else None
                )
                correction_text = self.correction_memory.build_prompt_addendum(
                    scenario_id=scenario_id,
                    stage=observation.workflow_stage,
                )
                system_prompt, user_prompt, response_format = build_runtime_request(
                    observation,
                    teacher_action=teacher_action,
                    correction_memory_text=correction_text,
                    strict=(mode == "strict"),
                )
                structured_mode = "backend_adaptive"
                raw_model_output = ""
                parse_result = None
                repair_retry_used = False
                teacher_override_used = False

                if (
                    mode == "lenient"
                    and repeated_no_progress >= 2
                    and teacher_action is not None
                ):
                    parse_result = self._teacher_override_parse_result(teacher_action)
                    raw_model_output = json.dumps(teacher_action)
                    teacher_override_used = True
                    repeated_no_progress = 0
                else:
                    response = self.backend.complete(
                        ModelRequest(
                            model_name=self.model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            structured_mode=structured_mode,
                            response_format=response_format,
                        )
                    )
                    raw_model_output = response.raw_text
                    parse_result = self.parser.parse(response.raw_text)

                    if (
                        mode == "lenient"
                        and parse_result.cleaned_action is None
                    ):
                        repair_retry_used = True
                        repair_system, repair_user = build_repair_request(
                            observation,
                            raw_bad_output=response.raw_text,
                            parse_error=parse_result.error or parse_result.parse_status,
                            teacher_action=teacher_action,
                        )
                        repair_response = self.backend.complete(
                            ModelRequest(
                                model_name=self.model_name,
                                system_prompt=repair_system,
                                user_prompt=repair_user,
                                structured_mode=structured_mode,
                                response_format=response_format,
                                temperature=0.0,
                            )
                        )
                        raw_model_output = repair_response.raw_text
                        parse_result = self.parser.parse(repair_response.raw_text)

                    if (
                        mode == "lenient"
                        and parse_result.cleaned_action is None
                        and teacher_action is not None
                    ):
                        parse_result = self._teacher_override_parse_result(teacher_action)
                        raw_model_output = json.dumps(teacher_action)
                        teacher_override_used = True

                assert parse_result is not None
                if parse_result.cleaned_action is None:
                    failure_reason = f"parse_failure:{parse_result.parse_status}"
                    observation_payload = observation.model_dump()
                    step_records.append(
                        StepRecord(
                            episode_id=episode_id,
                            scenario_id=scenario_id,
                            step_index=steps_taken + 1,
                            tick=observation.tick_count,
                            workflow_stage=observation.workflow_stage,
                            observation=observation_payload,
                            prompt_text=observation.prompt_text,
                            raw_model_output=raw_model_output,
                            parse_status=parse_result.parse_status,
                            normalization_applied=parse_result.repair_labels,
                            cleaned_action=None,
                            teacher_action=teacher_action,
                            reward=None,
                            cumulative_score=observation.final_score,
                            final_score_after_step=observation.final_score,
                            done=False,
                            next_prompt_text=None,
                            structured_mode_used=structured_mode,
                            repair_retry_used=repair_retry_used,
                            teacher_override_used=teacher_override_used,
                            failure_reason=parse_result.error,
                            failure_type=parse_result.parse_status,
                        )
                    )
                    break

                action = UnifiedIncidentAction(**parse_result.cleaned_action)
                step = self._unwrap_observation(env.step(action))
                observation_payload = observation.model_dump()
                step_records.append(
                    StepRecord(
                        episode_id=episode_id,
                        scenario_id=scenario_id,
                        step_index=steps_taken + 1,
                        tick=observation.tick_count,
                        workflow_stage=observation.workflow_stage,
                        observation=observation_payload,
                        prompt_text=observation.prompt_text,
                        raw_model_output=raw_model_output,
                        parse_status=parse_result.parse_status,
                        normalization_applied=parse_result.repair_labels,
                        cleaned_action=parse_result.cleaned_action,
                        teacher_action=teacher_action,
                        reward=step.reward,
                        cumulative_score=step.final_score,
                        final_score_after_step=step.final_score,
                        done=step.done,
                        next_prompt_text=step.prompt_text,
                        structured_mode_used=structured_mode,
                        repair_retry_used=repair_retry_used,
                        teacher_override_used=teacher_override_used,
                        failure_reason=None,
                        failure_type=None,
                    )
                )
                if step.reward is not None and step.reward <= 0 and parse_result.cleaned_action == previous_action:
                    repeated_no_progress += 1
                else:
                    repeated_no_progress = 0
                previous_action = parse_result.cleaned_action
                observation = step
                steps_taken += 1

            if failure_reason is None:
                success = bool(observation.done and observation.incident_resolved)
                final_score = observation.final_score
                if not success:
                    failure_reason = f"stopped:{observation.workflow_stage}"
            else:
                final_score = observation.final_score

        elapsed_s = round(time.perf_counter() - started, 4)
        return EpisodeRecord(
            run_id=str(uuid.uuid4()),
            episode_id=episode_id,
            scenario_id=scenario_id,
            difficulty=scenario["difficulty"],
            model_name=self.model_name,
            model_version=model_version,
            mode=mode,
            success=success,
            final_score=final_score,
            steps=steps_taken,
            failure_reason=failure_reason,
            elapsed_s=elapsed_s,
            json_valid_steps=sum(
                1 for step in step_records if step.parse_status in {"ok", "repaired", "teacher_override"}
            ),
            strict_schema_valid_steps=sum(
                1 for step in step_records if step.parse_status == "ok"
            ),
            teacher_override_count=sum(1 for step in step_records if step.teacher_override_used),
            repair_retry_count=sum(1 for step in step_records if step.repair_retry_used),
            security_subquest_completed=bool(
                step_records and any(
                    step.cleaned_action and step.cleaned_action.get("action_type") == "submit_security_fix"
                    for step in step_records
                )
            ),
            postmortem_completed=bool(
                step_records and any(
                    step.cleaned_action and step.cleaned_action.get("action_type") == "submit_postmortem"
                    for step in step_records
                )
            ),
            stopped_reason=(
                failure_reason.split(":", 1)[1]
                if failure_reason and failure_reason.startswith("stopped:")
                else failure_reason
            ),
            step_records=step_records,
        )

    def _env_context(self):
        if self.base_url is not None:
            return UnifiedIncidentEnv(base_url=self.base_url).sync()
        return _LocalEnvironmentContext(self.env_factory or UnifiedIncidentEnvironment)

    def _unwrap_observation(self, value):
        return value.observation if hasattr(value, "observation") else value

    def _teacher_override_parse_result(self, teacher_action: dict[str, Any]):
        from .types import ParseResult

        return ParseResult(
            parse_status="teacher_override",
            cleaned_action=teacher_action,
            repair_labels=["teacher_override"],
        )


class _LocalEnvironmentContext:
    """Simple context wrapper for direct environment instances."""

    def __init__(self, factory) -> None:
        self.factory = factory
        self.instance = None

    def __enter__(self):
        self.instance = self.factory()
        return self.instance

    def __exit__(self, exc_type, exc, tb):
        self.instance = None
        return False


def run_episode(
    *,
    model_name: str,
    scenario_id: str,
    mode: str,
    base_url: str | None,
    api_base_url: str,
    api_key: str,
    output_path: Path | None = None,
) -> EpisodeRecord:
    parser = StrictActionParser() if mode == "strict" else LenientActionAdapter()
    backend = OpenAICompatibleBackend(base_url=api_base_url, api_key=api_key)
    runner = EpisodeRunner(
        backend=backend,
        parser=parser,
        model_name=model_name,
        base_url=base_url,
    )
    record = runner.run(scenario_id=scenario_id, mode=mode)
    if output_path is not None:
        TrajectoryStore(output_path).append_episode(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--mode", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--base-url", default=UnifiedIncidentEnv.DEFAULT_BASE_URL)
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
        default="outputs/trainer/episodes.jsonl",
    )
    args = parser.parse_args()

    record = run_episode(
        model_name=args.model,
        scenario_id=args.scenario,
        mode=args.mode,
        base_url=args.base_url,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        output_path=Path(args.output),
    )
    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
