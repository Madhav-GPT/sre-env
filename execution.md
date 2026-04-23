# How To Run (v2)

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 2. Start the environment

```bash
source .venv/bin/activate
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

## 3. Manual API smoke test

```bash
curl -X POST http://127.0.0.1:8000/reset -H 'content-type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8000/step -H 'content-type: application/json' -d '{"action":{"action_type":"query_deploys","service":"worker"}}'
```

## 4. Run inference

```bash
source .venv/bin/activate

export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct:novita"
export ENV_BASE_URL="http://127.0.0.1:8000"

python inference.py
```

## 5. Verification

```bash
source .venv/bin/activate
pytest unified_incident_env/tests -q
openenv validate .
```

## 6. Reward semantics

- queries reveal evidence but do not directly mint positive breadcrumb reward
- remediation actions change the world state
- `run_check` verifies recovery explicitly
- `declare_resolved` succeeds only after objective checks pass

Public benchmark score is deterministic and separate from the per-step training reward.
