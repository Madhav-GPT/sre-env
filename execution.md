# How To Run

This is the simple version.

## 1. Setup

Open a terminal in the repo:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/daksh/my-openenv
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 2. Start the environment

Keep this running in terminal 1:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/daksh/my-openenv
source .venv/bin/activate
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

## 3. Run inference with your HF token

Run this in terminal 2:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/daksh/my-openenv
source .venv/bin/activate

export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct:novita"
export ENV_BASE_URL="http://127.0.0.1:8000"

python inference.py
```

That runs all 3 tasks.

## 4. Required env vars

This repo uses:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `ENV_BASE_URL`

Also present for competition checklist compatibility:

- `LOCAL_IMAGE_NAME`

You do not need `LOCAL_IMAGE_NAME` for the normal run above.

## 5. What the script prints

`inference.py` prints lines like:

```text
[START] ...
[STEP] ...
[END] ...
```

The final line includes the task score.

## 6. Before submitting

Run these:

```bash
cd /Users/madhav_189/Documents/meta_hackathon/daksh/my-openenv
source .venv/bin/activate
pytest unified_incident_env/tests -q
openenv validate .
```

Expected right now:

- `54 passed`
- `Ready for multi-mode deployment`

## 7. Important

- The competition wants scores strictly between `0.0` and `1.0`.
- This repo is set up for that now.
- If you change code, push it and redeploy the Hugging Face Space before submitting.
