"""Import-patch adapter for OpenClaw-RL's `terminal-rl/generate.py`.

STUB — filled in Friday when the OpenClaw-RL venv is set up.

The shape is minimal: OpenClaw-RL's `terminal-rl/generate.py` does
`from env_client import create_env_client`. All we need is to redirect that
import to our client. Two options, pick one Friday:

  Option A: monkey-patch via PYTHONPATH + shim module
    export PYTHONPATH="/path/to/sre-enginnerllm:$PYTHONPATH"
    mkdir -p /tmp/openclaw_shim && cd /tmp/openclaw_shim
    cat > env_client.py <<'PY'
    from openclaw_integration.sre_env_client import create_env_client
    PY
    export PYTHONPATH="/tmp/openclaw_shim:$PYTHONPATH"

  Option B: patch generate.py directly
    sed -i 's|from env_client import create_env_client|from openclaw_integration.sre_env_client import create_env_client|' \
        /path/to/OpenClaw-RL/terminal-rl/generate.py

Option A is reversible and cleaner. Option B is one line and survives a
pip install -e.

This file is intentionally empty beyond this docstring to keep the shim
surface area tiny. When Friday work begins, the actual adapter (if any is
needed beyond the import swap) lives here.
"""
