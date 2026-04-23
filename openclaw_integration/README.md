# OpenClaw-RL integration — sre-gym shim

Plugs `sre-gym` into OpenClaw-RL's training loop without forking OpenClaw-RL.
Three artifacts:

- `pool_server.py` — FastAPI HTTP server speaking OpenClaw's lease-based
  contract (`/allocate /reset /exec_tool /evaluate /close`). Wraps
  `UnifiedIncidentEnvironment` behind per-lease `asyncio.Lock`s.
- `sre_env_client.py` — Drop-in replacement for OpenClaw-RL
  `terminal-rl/env_client.py`. Same method signatures.
- `generate_with_sre.py` — Planned import-patch wrapper for
  `terminal-rl/generate.py` (stub — filled in Friday when the OpenClaw-RL
  venv is set up).

## Quick start

```bash
# 1. Launch the pool server
source .venv/bin/activate
uvicorn openclaw_integration.pool_server:app --host 0.0.0.0 --port 8100

# 2. Smoke-test the lifecycle from another shell
curl -sf http://127.0.0.1:8100/healthz | jq
curl -s -X POST http://127.0.0.1:8100/allocate \
     -H 'content-type: application/json' \
     -d '{"task_key": "gateway_auth_rollout"}'
```

## Wiring into OpenClaw-RL

In the OpenClaw-RL repo, after creating a fresh venv per their instructions,
point the rollout agent at our server:

```bash
export ENV_SERVER_URL=http://127.0.0.1:8100
```

Then patch one import in `OpenClaw-RL/terminal-rl/generate.py`:

```diff
- from env_client import create_env_client
+ import sys; sys.path.insert(0, "/path/to/sre-enginnerllm")
+ from openclaw_integration.sre_env_client import create_env_client
```

No other OpenClaw-RL source files need to change. The
`run_qwen35_4b_openclaw_rl.sh` launch script works as-is after that.

## Task keys (scenarios)

- `worker_deploy_cascade` (easy)
- `db_config_rollout` (medium)
- `gateway_auth_rollout` (hard)

## Lifecycle contract

```
allocate(task_key)                    -> {ok: true, lease_id}
reset(lease_id, task_meta, run_ctx)   -> {ok: true, observation: "<json>"}
exec_tool(lease_id, tool_call)        -> {ok: true, observation: "<json>"}
evaluate(lease_id)                    -> {ok: true, score: float}
close(lease_id)                       -> {ok: true}
```

- `task_meta.scenario_id` takes precedence over `task_key` at reset time if
  set (useful for procgen Friday).
- `tool_call.name` maps directly to `UnifiedIncidentAction.action_type`.
- `tool_call.arguments` is the kwargs dict (service, metric, check_name,
  hypothesis).
- An invalid action is returned as an observation `{"error": "...",
  "tool_call": {...}}` rather than raising — training gets the negative
  signal without crashing the rollout.

## Lease TTL / reaper

- `POOL_SERVER_LEASE_TTL_S` (default 600s) — lease idle timeout.
- `POOL_SERVER_REAPER_PERIOD` (default 30s) — reaper tick period.

Reaper runs in lifespan background task; evicts idle leases so long
training runs don't leak env instances.
