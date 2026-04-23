#!/usr/bin/env bash
# sre-gym end-to-end demo.
# Spins up the env (or reuses a running one), solves each of the 3 scenarios
# with the baseline policy, records runbooks, shows the artefacts.
#
# Requires: python3.10+, docker (for the HF-Space-equivalent image) OR the
# repo's .venv. Defaults to .venv if present.

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8013}"
URL="http://127.0.0.1:${PORT}"
PY="${PYTHON:-.venv/bin/python}"
RUNBOOK_DIR="skill/verified-runbooks"

banner() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()     { printf '\033[0;32m  ✓ %s\033[0m\n' "$*"; }

banner "0 / preflight"
if [[ ! -x "$PY" ]]; then
  echo "  note: $PY not found, falling back to system python3" >&2
  PY="python3"
fi
"$PY" -c "import unified_incident_env" 2>/dev/null || {
  echo "  error: unified_incident_env not importable; run 'pip install -e .' first" >&2
  exit 1
}
ok "python + package ready"

banner "1 / start env"
if curl -sf "$URL/health" > /dev/null 2>&1; then
  ok "env already running on $URL"
  SERVER_STARTED=0
else
  "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT" > /tmp/sre_gym_demo.log 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED=1
  for _ in $(seq 1 20); do
    if curl -sf "$URL/health" > /dev/null 2>&1; then break; fi
    sleep 0.3
  done
  curl -sf "$URL/health" > /dev/null || { echo "  error: env failed to start" >&2; cat /tmp/sre_gym_demo.log >&2; exit 1; }
  ok "env started on $URL (pid $SERVER_PID)"
fi
trap '[[ ${SERVER_STARTED:-0} -eq 1 ]] && kill ${SERVER_PID:-0} 2>/dev/null || true' EXIT

banner "2 / available scenarios"
SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py list

banner "3 / clear prior runbooks (demo starts cold)"
rm -f "$RUNBOOK_DIR"/*.md
ok "runbook directory cleared"

for scenario in worker_deploy_cascade db_config_rollout gateway_auth_rollout; do
  banner "4 / solve: $scenario"
  SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py solve "$scenario"
  SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py record-runbook "$scenario"
done

banner "5 / verified runbooks now on disk"
ls -1 "$RUNBOOK_DIR"/*.md | sed 's|^|  |'

banner "6 / re-solve easy scenario — runbook is loaded this time"
SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py solve worker_deploy_cascade | tail -4

banner "done"
echo "  install the skill globally:   ln -s \"$PWD/skill\" \"\$HOME/.claude/skills/sre-gym\""
echo "  env log:                      /tmp/sre_gym_demo.log"
echo "  runbooks:                     $RUNBOOK_DIR/"
