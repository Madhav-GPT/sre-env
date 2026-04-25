#!/usr/bin/env bash
# sre-gym end-to-end demo (round-2 catalogue, 12 templates).
# Exercises one scenario per failure *category* at seed=42, records runbooks,
# and shows the artefacts.
#
# Categories cover the round-2 catalogue:
#   deploy   -> worker_deploy_cascade        (classic deploy regression)
#   config   -> db_config_rollout            (config-vs-code disambiguation)
#   auth     -> auth_token_expiry            (cross-service credential propagation)
#   data     -> migration_lock               (lock contention without crash)

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8013}"
URL="http://127.0.0.1:${PORT}"
PY="${PYTHON:-.venv/bin/python}"
RUNBOOK_DIR="skill/verified-runbooks"
SEED="${SEED:-42}"

SCENARIOS=(
  "worker_deploy_cascade"
  "db_config_rollout"
  "auth_token_expiry"
  "migration_lock"
)

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
ok "python + package ready (seed=$SEED, ${#SCENARIOS[@]} scenarios)"

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
  curl -sf "$URL/health" > /dev/null || {
    echo "  error: env failed to start" >&2
    cat /tmp/sre_gym_demo.log >&2
    exit 1
  }
  ok "env started on $URL (pid $SERVER_PID)"
fi
trap '[[ ${SERVER_STARTED:-0} -eq 1 ]] && kill ${SERVER_PID:-0} 2>/dev/null || true' EXIT

banner "2 / catalogue summary"
SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py list | head -20
echo "  (full catalogue: 12 templates × 6 procgen variants = 72 scenarios)"

banner "3 / clear prior verified runbooks (drafts preserved)"
mkdir -p "$RUNBOOK_DIR"
find "$RUNBOOK_DIR" -name "*.md" -exec grep -l "status: verified" {} \; 2>/dev/null | xargs -r rm -f || true
ok "verified runbooks purged; drafts preserved"

for scenario in "${SCENARIOS[@]}"; do
  banner "4 / solve: $scenario (seed=$SEED)"
  SRE_GYM_URL="$URL" SRE_GYM_SEED="$SEED" \
    "$PY" skill/tools/sre_gym_client.py solve "$scenario"
  SRE_GYM_URL="$URL" \
    "$PY" skill/tools/sre_gym_client.py record-runbook "$scenario"
done

banner "5 / runbooks now on disk (12 total: 3 verified + 9 draft)"
ls -1 "$RUNBOOK_DIR"/*.md | sed 's|^|  |'

banner "6 / re-solve worker_deploy_cascade — runbook is loaded this time"
SRE_GYM_URL="$URL" "$PY" skill/tools/sre_gym_client.py solve worker_deploy_cascade | tail -4

banner "done"
echo "  install the skill globally:   ln -s \"$PWD/skill\" \"\$HOME/.claude/skills/sre-gym\""
echo "  env log:                      /tmp/sre_gym_demo.log"
echo "  runbooks:                     $RUNBOOK_DIR/"
echo "  full catalogue (12 × 6 procgen variants): curl -s $URL/tasks | jq '.scenarios | length'"
