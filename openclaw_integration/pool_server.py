"""FastAPI pool server exposing sre-gym in OpenClaw-RL's lease-based contract.

OpenClaw-RL's rollout agent drives an env with this lifecycle per episode:

    allocate(task_key)  -> {lease_id}
    reset(lease_id, task_meta, run_ctx)
    exec_tool(lease_id, tool_call)  -> observation_string   # repeated
    evaluate(lease_id)              -> score
    close(lease_id)

We wrap a `UnifiedIncidentEnvironment` instance per lease. Lease state is
guarded by per-lease `asyncio.Lock` so 8-way concurrent rollouts on the same
server stay consistent. Idle leases are reaped after LEASE_TTL_S seconds.

Run standalone:
    uvicorn openclaw_integration.pool_server:app --host 0.0.0.0 --port 8100

Env vars:
    POOL_SERVER_LEASE_TTL_S   default 600
    POOL_SERVER_REAPER_PERIOD default 30
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Make the sibling package importable when launched via uvicorn from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unified_incident_env.models import UnifiedIncidentAction  # noqa: E402
from unified_incident_env.server.challenge import SCENARIOS  # noqa: E402
from unified_incident_env.server.environment import UnifiedIncidentEnvironment  # noqa: E402

logger = logging.getLogger("sre_gym.pool_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LEASE_TTL_S = float(os.getenv("POOL_SERVER_LEASE_TTL_S", "600"))
REAPER_PERIOD_S = float(os.getenv("POOL_SERVER_REAPER_PERIOD", "30"))


@dataclass
class Lease:
    lease_id: str
    task_key: str
    env: UnifiedIncidentEnvironment
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_touch: float = field(default_factory=time.time)
    reset_done: bool = False
    final_score: float | None = None

    def touch(self) -> None:
        self.last_touch = time.time()


class AllocateRequest(BaseModel):
    task_key: str
    request_id: str | None = None


class LeaseRequest(BaseModel):
    lease_id: str


class ResetRequest(BaseModel):
    lease_id: str
    task_meta: dict[str, Any] = Field(default_factory=dict)
    run_ctx: dict[str, Any] = Field(default_factory=dict)
    task_timeouts: dict[str, Any] | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExecToolRequest(BaseModel):
    lease_id: str
    tool_call: ToolCall


class LeasePool:
    def __init__(self) -> None:
        self._leases: dict[str, Lease] = {}
        self._dict_lock = asyncio.Lock()

    async def allocate(self, task_key: str) -> Lease:
        if task_key not in SCENARIOS:
            raise ValueError(f"Unknown task_key {task_key!r}; known: {list(SCENARIOS)}")
        env = UnifiedIncidentEnvironment()
        lease = Lease(lease_id=str(uuid.uuid4()), task_key=task_key, env=env)
        async with self._dict_lock:
            self._leases[lease.lease_id] = lease
        logger.info("allocate: lease=%s task=%s", lease.lease_id, task_key)
        return lease

    async def get(self, lease_id: str) -> Lease:
        async with self._dict_lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            raise KeyError(f"Unknown lease {lease_id}")
        lease.touch()
        return lease

    async def close(self, lease_id: str) -> bool:
        async with self._dict_lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        logger.info("close: lease=%s task=%s", lease_id, lease.task_key)
        return True

    async def reap(self) -> int:
        now = time.time()
        stale: list[str] = []
        async with self._dict_lock:
            for lease_id, lease in list(self._leases.items()):
                if now - lease.last_touch > LEASE_TTL_S:
                    stale.append(lease_id)
            for lease_id in stale:
                self._leases.pop(lease_id, None)
        if stale:
            logger.info("reaper: evicted %d stale lease(s)", len(stale))
        return len(stale)

    def active_count(self) -> int:
        return len(self._leases)


pool = LeasePool()


async def _reaper_loop() -> None:
    while True:
        try:
            await pool.reap()
        except Exception:
            logger.exception("reaper loop tick failed")
        await asyncio.sleep(REAPER_PERIOD_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_reaper_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="sre-gym OpenClaw pool server", lifespan=lifespan)


def _observation_string(obs: Any, *, reward: float | None = None) -> str:
    """Render a UnifiedIncidentObservation as the single string OpenClaw
    rollout agents expect from exec_tool."""
    payload = {
        "tick": obs.tick_count,
        "workflow_stage": obs.workflow_stage,
        "last_action_result": obs.last_action_result,
        "tool_output": obs.tool_output,
        "failure_type": obs.failure_type,
        "why_failed": obs.why_failed,
        "loop_warning": obs.loop_warning,
        "reward": reward,
        "checks": [{"name": c.name, "passed": c.passed} for c in obs.checks],
        "active_alerts": [{"service": a.service, "severity": a.severity, "message": a.message} for a in obs.active_alerts],
        "noise_alerts": [{"service": a.service, "severity": a.severity, "message": a.message} for a in obs.noise_alerts],
        "service_health": {name: s.status for name, s in obs.service_health.items()},
        "allowed_actions": obs.allowed_actions,
        "required_fields_by_action": obs.required_fields_by_action,
        "blast_radius": obs.blast_radius,
        "final_score": obs.final_score,
        "done": obs.done,
        "prompt_text": obs.prompt_text,
    }
    return json.dumps(payload, separators=(",", ":"))


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "active_leases": pool.active_count(), "scenarios": list(SCENARIOS.keys())}


@app.post("/allocate")
async def allocate(request: AllocateRequest) -> dict[str, Any]:
    try:
        lease = await pool.allocate(request.task_key)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "lease_id": lease.lease_id, "task_key": lease.task_key, "request_id": request.request_id}


@app.post("/heartbeat")
async def heartbeat(request: LeaseRequest) -> dict[str, Any]:
    try:
        await pool.get(request.lease_id)
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    try:
        lease = await pool.get(request.lease_id)
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}
    async with lease.lock:
        scenario_id = request.task_meta.get("scenario_id") or lease.task_key
        obs = lease.env.reset(scenario_id=scenario_id)
        lease.reset_done = True
        lease.final_score = None
    return {"ok": True, "observation": _observation_string(obs)}


@app.post("/exec_tool")
async def exec_tool(request: ExecToolRequest) -> dict[str, Any]:
    try:
        lease = await pool.get(request.lease_id)
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}
    if not lease.reset_done:
        return {"ok": False, "error": "reset has not been called for this lease"}

    action_kwargs = {"action_type": request.tool_call.name, **request.tool_call.arguments}
    try:
        action = UnifiedIncidentAction(**action_kwargs)
    except Exception as exc:
        # Return the validation error to the rollout agent as a no-op
        # observation so training sees the failure signal without crashing.
        return {"ok": True, "observation": json.dumps({"error": f"invalid action: {exc}", "tool_call": request.tool_call.model_dump()})}

    async with lease.lock:
        obs = lease.env.step(action)
        lease.final_score = float(obs.final_score)
    return {"ok": True, "observation": _observation_string(obs, reward=float(obs.reward))}


@app.post("/evaluate")
async def evaluate(request: LeaseRequest) -> dict[str, Any]:
    try:
        lease = await pool.get(request.lease_id)
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}
    score = lease.final_score if lease.final_score is not None else float(lease.env.state.final_score)
    return {"ok": True, "score": score}


@app.post("/close")
async def close(request: LeaseRequest) -> dict[str, Any]:
    closed = await pool.close(request.lease_id)
    if not closed:
        return {"ok": False, "error": f"Unknown lease {request.lease_id}"}
    return {"ok": True}
