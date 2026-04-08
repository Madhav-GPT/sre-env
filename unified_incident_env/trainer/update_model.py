"""Pluggable model updater backends for the session loop."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

from .types import UpdateRequest, UpdateResult


class NoOpUpdater:
    """Updater that records an update request without changing weights."""

    backend_name = "noop"

    def update(self, request: UpdateRequest) -> UpdateResult:
        output_dir = Path(request.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / f"update_{request.update_index:02d}_request.json"
        result_path = output_dir / f"update_{request.update_index:02d}_result.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

        result = UpdateResult(
            update_index=request.update_index,
            phase_name=request.phase_name,
            updater_backend="noop",
            model_before=request.model_before,
            model_after=request.model_before,
            status="noop",
            episodes_used=request.episodes_used,
            datasets_used=request.datasets_used,
            artifact_paths=[str(request_path), str(result_path)],
            notes="NoOpUpdater recorded the update request but did not change weights.",
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


class ExternalCommandUpdater:
    """Updater that delegates weight updates to an external command."""

    backend_name = "external_command"

    def __init__(self, command_template: str) -> None:
        self.command_template = command_template

    def _command_for_current_python(self) -> str:
        executable = shlex.quote(sys.executable)
        if self.command_template.startswith("python "):
            return executable + self.command_template[len("python") :]
        if self.command_template.startswith("python3 "):
            return executable + self.command_template[len("python3") :]
        return self.command_template

    def update(self, request: UpdateRequest) -> UpdateResult:
        output_dir = Path(request.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / f"update_{request.update_index:02d}_request.json"
        result_path = output_dir / f"update_{request.update_index:02d}_result.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

        command = self._command_for_current_python().format(
            output_dir=str(output_dir),
            request_path=str(request_path),
            model_before=request.model_before,
        )
        completed = subprocess.run(
            command,
            shell=True,
            cwd=output_dir,
            capture_output=True,
            text=True,
        )
        external_result_path = output_dir / f"update_{request.update_index:02d}_external_result.json"
        status = "completed" if completed.returncode == 0 else "failed"
        model_after = str(output_dir) if completed.returncode == 0 else request.model_before
        notes = (
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        ).strip()
        artifact_paths = [str(request_path), str(result_path)]

        if completed.returncode == 0 and external_result_path.exists():
            payload = json.loads(external_result_path.read_text(encoding="utf-8"))
            status = payload.get("status", status)
            model_after = payload.get("model_after", model_after)
            notes = payload.get("notes", notes)
            artifact_paths.extend(payload.get("artifact_paths", []))
            artifact_paths.append(str(external_result_path))
        elif completed.returncode == 0:
            stdout_text = completed.stdout.strip()
            if stdout_text.startswith("{") and stdout_text.endswith("}"):
                try:
                    payload = json.loads(stdout_text)
                    status = payload.get("status", status)
                    model_after = payload.get("model_after", model_after)
                    notes = payload.get("notes", notes)
                    artifact_paths.extend(payload.get("artifact_paths", []))
                except Exception:
                    pass

        result = UpdateResult(
            update_index=request.update_index,
            phase_name=request.phase_name,
            updater_backend="external_command",
            model_before=request.model_before,
            model_after=model_after if status == "completed" else request.model_before,
            status=status,
            episodes_used=request.episodes_used,
            datasets_used=request.datasets_used,
            artifact_paths=artifact_paths,
            notes=notes,
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


class OpenAIFineTuneUpdater:
    """Optional hosted fine-tune backend for research mode only."""

    backend_name = "openai_finetune"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        client: OpenAI | None = None,
        poll_interval_s: float = 5.0,
        timeout_s: float = 1800.0,
    ) -> None:
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
        )
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    def update(self, request: UpdateRequest) -> UpdateResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / f"update_{request.update_index:02d}_request.json"
        result_path = output_dir / f"update_{request.update_index:02d}_result.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

        if not request.training_file:
            result = UpdateResult(
                update_index=request.update_index,
                phase_name=request.phase_name,
                updater_backend="openai_finetune",
                model_before=request.model_before,
                model_after=request.model_before,
                status="failed",
                episodes_used=request.episodes_used,
                datasets_used=request.datasets_used,
                artifact_paths=[str(request_path), str(result_path)],
                notes="training_file is required for openai_finetune updater",
            )
            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result

        training_path = Path(request.training_file)
        training_file = None
        try:
            with training_path.open("rb") as handle:
                training_file = self.client.files.create(
                    file=handle,
                    purpose="fine-tune",
                )
            job = self.client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model=request.model_before,
                suffix=request.suffix,
            )
            started = time.monotonic()
            status = getattr(job, "status", "queued")
            final_job = job
            while status not in {"succeeded", "failed", "cancelled"}:
                if time.monotonic() - started > self.timeout_s:
                    status = "failed"
                    break
                time.sleep(self.poll_interval_s)
                final_job = self.client.fine_tuning.jobs.retrieve(job.id)
                status = getattr(final_job, "status", "queued")

            model_after = getattr(final_job, "fine_tuned_model", None) or request.model_before
            result_status = "completed" if status == "succeeded" else "failed"
            notes = f"job_id={job.id} status={status}"
        except Exception as exc:
            model_after = request.model_before
            result_status = "failed"
            notes = f"{type(exc).__name__}: {exc}"

        result = UpdateResult(
            update_index=request.update_index,
            phase_name=request.phase_name,
            updater_backend="openai_finetune",
            model_before=request.model_before,
            model_after=model_after,
            status=result_status,
            episodes_used=request.episodes_used,
            datasets_used=request.datasets_used,
            artifact_paths=[str(request_path), str(result_path)],
            notes=notes,
        )
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


def build_updater(
    backend_name: str,
    *,
    runtime_mode: str = "competition",
    command_template: str | None = None,
    openai_base_url: str | None = None,
    api_key: str | None = None,
):
    """Return the configured updater backend."""
    if backend_name == "noop":
        return NoOpUpdater()
    if backend_name == "external_command":
        if not command_template:
            command_template = (
                "python -m unified_incident_env.trainer.train_external "
                "--request {request_path}"
            )
        return ExternalCommandUpdater(command_template)
    if backend_name == "openai_finetune":
        if runtime_mode != "research":
            raise ValueError("openai_finetune updater is allowed only in research mode")
        if not openai_base_url or not api_key:
            raise ValueError("openai_base_url and api_key are required for openai_finetune")
        return OpenAIFineTuneUpdater(base_url=openai_base_url, api_key=api_key)
    raise ValueError(f"Unknown updater backend {backend_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--backend", choices=["noop", "external_command", "openai_finetune"], default="noop")
    parser.add_argument("--runtime-mode", choices=["competition", "research"], default="competition")
    parser.add_argument("--command-template")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    request = UpdateRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    updater = build_updater(
        args.backend,
        runtime_mode=args.runtime_mode,
        command_template=args.command_template,
        openai_base_url=args.openai_base_url,
        api_key=args.api_key,
    )
    result = updater.update(request)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
