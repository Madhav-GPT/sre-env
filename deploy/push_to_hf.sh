#!/usr/bin/env bash
# Deploy this repo to a Hugging Face Space (Docker SDK).
#
# Required:
#   HF_TOKEN      write-scoped HF access token
#   HF_SPACE_ID   e.g. yourname/sre-gym  (create it at huggingface.co/new-space
#                 first, SDK=Docker, or let this script try to create it)
#
# Usage:
#   HF_TOKEN=hf_xxx HF_SPACE_ID=yourname/sre-gym ./deploy/push_to_hf.sh
#
# After a successful push, verify from a different network:
#   curl https://${space_subdomain}.hf.space/health
#   curl https://${space_subdomain}.hf.space/tasks | jq '.scenarios[].difficulty'

set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_SPACE_ID:?HF_SPACE_ID is required, e.g. yourname/sre-gym}"

if ! command -v huggingface-cli > /dev/null; then
  echo "error: huggingface-cli not installed. pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "== syncing openenv.yaml with HF_SPACE_ID =="
python3 - <<PY
import pathlib, re
path = pathlib.Path("openenv.yaml")
text = path.read_text()
text = re.sub(r"^  space_id:.*$", f"  space_id: $HF_SPACE_ID", text, flags=re.M)
path.write_text(text)
print(f"openenv.yaml space_id -> $HF_SPACE_ID")
PY

echo "== ensuring the space exists (idempotent) =="
huggingface-cli repo create "$HF_SPACE_ID" \
  --type space \
  --space_sdk docker \
  --token "$HF_TOKEN" \
  --yes 2>&1 | grep -v "already created" || true

echo "== uploading repo =="
huggingface-cli upload "$HF_SPACE_ID" . \
  --repo-type space \
  --token "$HF_TOKEN" \
  --commit-message "deploy sre-gym v2 (easy/medium/hard scenarios)"

subdomain="$(echo "$HF_SPACE_ID" | tr '/' '-')"
echo
echo "== deployment kicked off =="
echo "   Logs:     https://huggingface.co/spaces/$HF_SPACE_ID"
echo "   Public:   https://$subdomain.hf.space"
echo
echo "== verify from a different network (phone hotspot) =="
echo "   curl https://$subdomain.hf.space/health"
echo "   curl https://$subdomain.hf.space/tasks | jq '.scenarios[].difficulty'"
