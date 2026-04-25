# Coliseum — parallel-rollout pool server for sre-gym Triage

Coliseum exposes the Triage tier of sre-gym through a lease-based HTTP
contract so a GRPO trainer's parallel-rollout side can drive the env without
holding an in-process `UnifiedIncidentEnvironment` per worker. The shape is
the standard lease-pool pattern (`allocate → reset → exec_tool* → evaluate →
close`) used by every parallel-rollout RL framework — Coliseum is just the
sre-gym implementation of that contract.

Two artifacts:

- `coliseum/server.py` — FastAPI HTTP server. Wraps one
  `UnifiedIncidentEnvironment` per lease behind an `asyncio.Lock` so 8-way
  concurrent rollouts on the same process stay consistent.
- `coliseum/client.py` — `ArenaClient`, an async HTTP client that calls the
  server over `httpx` with retry/backoff per route. `create_arena_client()`
  reads `COLISEUM_BASE_URL` and constructs one.

## Quick start

```bash
# 1. Launch the pool server
source .venv/bin/activate
uvicorn coliseum.server:app --host 0.0.0.0 --port 8100

# 2. Smoke-test the lifecycle from another shell
curl -sf http://127.0.0.1:8100/healthz | jq
curl -s -X POST http://127.0.0.1:8100/allocate \
     -H 'content-type: application/json' \
     -d '{"task_key": "gateway_auth_rollout"}'
```

## Wiring into a GRPO trainer

Set `COLISEUM_BASE_URL` for the rollout workers and call
`create_arena_client()` from your trainer's env-driver module. The contract
mirrors the standard lease-pool shape — point your existing pool client at
Coliseum, or use `ArenaClient` directly:

```python
import asyncio
from coliseum import create_arena_client

async def rollout_one(task_key: str) -> float:
    client = create_arena_client()
    lease = await client.allocate(task_key)
    lease_id = lease["lease_id"]
    try:
        await client.reset(lease_id, task_meta={"scenario_id": task_key}, run_ctx={})
        # ... drive exec_tool() until done ...
        return await client.evaluate(lease_id)
    finally:
        await client.close(lease_id)

asyncio.run(rollout_one("gateway_auth_rollout"))
```

## Lifecycle contract

```
allocate(task_key)                    -> {ok: true, lease_id}
reset(lease_id, task_meta, run_ctx)   -> {ok: true, observation: "<json>"}
exec_tool(lease_id, tool_call)        -> {ok: true, observation: "<json>"}
evaluate(lease_id)                    -> {ok: true, score: float}
close(lease_id)                       -> {ok: true}
```

- `task_meta.scenario_id` takes precedence over `task_key` at reset time
  if set (lets the rollout side pin a procgen variant per attempt).
- `tool_call.name` maps directly to `UnifiedIncidentAction.action_type`.
- `tool_call.arguments` is the kwargs dict (`service`, `metric`,
  `check_name`, `hypothesis`).
- An invalid action is returned as `{"error": "...", "tool_call": {...}}`
  inside `observation` rather than raising — training gets the negative
  signal without the rollout crashing.

## Task keys

Any `scenario_id` registered in `unified_incident_env/server/challenge.py`
is a valid `task_key`. The 12 base templates plus 5 procgen variants per
template gives 72 task keys; `GET /healthz` returns the full list.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `COLISEUM_BASE_URL` | (required) | Server URL for `create_arena_client()` |
| `COLISEUM_LEASE_TTL_S` | `600` | Idle-lease eviction window |
| `COLISEUM_REAPER_PERIOD_S` | `30` | Reaper tick period |
| `COLISEUM_HTTP_TIMEOUT_S` | `30` | Per-request HTTP timeout |
| `COLISEUM_HTTP_MAX_RETRIES` | `10` | Default retry budget |
| `COLISEUM_ALLOCATE_MAX_RETRIES` | `10` | Allocate-route retry budget |
| `COLISEUM_EXEC_TOOL_MAX_RETRIES` | `3` | exec_tool-route retry budget |
| `COLISEUM_EVALUATE_MAX_RETRIES` | `1` | Evaluate-route retry budget |
| `COLISEUM_CLOSE_MAX_RETRIES` | `3` | Close-route retry budget |

## Lease reaper

A background task on the FastAPI lifespan reaps leases idle longer than
`COLISEUM_LEASE_TTL_S`. Long training runs don't leak env instances; if a
rollout worker dies mid-episode its lease is evicted on the next reaper
tick.
