"""Drop-in replacement for OpenClaw-RL `terminal-rl/env_client.py`.

Interface matches `TerminalEnvClient` (allocate / heartbeat / reset /
exec_tool / evaluate / close) so OpenClaw-RL's rollout agent can swap imports
with one line.

Standalone (no slime dep) — uses httpx directly. To use slime's retrying
post() helper instead, replace `_post` with `slime.utils.http_utils.post`.

Env vars:
    ENV_SERVER_URL                   required, e.g. http://127.0.0.1:8100
    ENV_HTTP_MAX_RETRIES             default 10
    ENV_ALLOCATE_MAX_RETRIES         default 10
    ENV_EVALUATE_MAX_RETRIES         default 1
    ENV_CLOSE_MAX_RETRIES            default 3
    ENV_EXEC_TOOL_MAX_RETRIES        default 3
    ENV_HTTP_TIMEOUT_S               default 30
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def create_env_client() -> "SreEnvClient":
    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")
    return SreEnvClient(env_server_url)


async def _post(
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
    timeout_s: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # retry all transport errors
            last_exc = exc
            wait = min(2 ** attempt * 0.25, 5.0)
            logger.debug("POST %s failed (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)
            await asyncio.sleep(wait)
    raise RuntimeError(f"POST {url} failed after {max_retries} retries: {last_exc}")


class SreEnvClient:
    """OpenClaw-RL-shaped client for the sre-gym pool server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_max_retries = int(os.getenv("ENV_HTTP_MAX_RETRIES", "10"))
        self.allocate_max_retries = int(os.getenv("ENV_ALLOCATE_MAX_RETRIES", "10"))
        self.evaluate_max_retries = int(os.getenv("ENV_EVALUATE_MAX_RETRIES", "1"))
        self.close_max_retries = int(os.getenv("ENV_CLOSE_MAX_RETRIES", "3"))
        self.exec_tool_max_retries = int(os.getenv("ENV_EXEC_TOOL_MAX_RETRIES", "3"))
        self.timeout_s = float(os.getenv("ENV_HTTP_TIMEOUT_S", "30"))

    async def allocate(self, task_key: str, request_id: str | None = None) -> dict[str, Any]:
        out = await _post(
            f"{self.base_url}/allocate",
            {"task_key": task_key, "request_id": request_id},
            max_retries=self.allocate_max_retries,
            timeout_s=self.timeout_s,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await _post(
            f"{self.base_url}/heartbeat",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
            timeout_s=self.timeout_s,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"heartbeat failed: {out}")

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = await _post(
            f"{self.base_url}/reset",
            {
                "lease_id": lease_id,
                "task_meta": task_meta,
                "run_ctx": run_ctx,
                "task_timeouts": task_timeouts,
            },
            max_retries=self.default_max_retries,
            timeout_s=self.timeout_s,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"reset failed: {out}")
        return out

    async def exec_tool(self, lease_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
        out = await _post(
            f"{self.base_url}/exec_tool",
            {
                "lease_id": lease_id,
                "tool_call": {"name": tool_name, "arguments": arguments},
            },
            max_retries=self.exec_tool_max_retries,
            timeout_s=self.timeout_s,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"exec_tool failed: {out}")
        return str(out.get("observation", ""))

    async def evaluate(self, lease_id: str) -> float:
        out = await _post(
            f"{self.base_url}/evaluate",
            {"lease_id": lease_id},
            max_retries=self.evaluate_max_retries,
            timeout_s=self.timeout_s,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"evaluate failed: {out}")
        return float(out.get("score", 0.0))

    async def close(self, lease_id: str) -> None:
        try:
            out = await _post(
                f"{self.base_url}/close",
                {"lease_id": lease_id},
                max_retries=self.close_max_retries,
                timeout_s=self.timeout_s,
            )
        except Exception as exc:
            if "Unknown lease" in str(exc):
                logger.debug("close(%s): lease already gone", lease_id)
                return
            raise
        if not out.get("ok", False):
            error_msg = str(out.get("error", ""))
            if "Unknown" in error_msg and "lease" in error_msg.lower():
                logger.debug("close(%s): lease already gone", lease_id)
                return
            raise RuntimeError(f"close failed: {out}")
