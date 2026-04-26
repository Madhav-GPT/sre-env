# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # sre-gym Triage — Qwen2.5-7B SFT → GRPO training notebook
#
# **Target:** A100 80GB · ~2-3h wall-clock · ~$5-8 in HF compute credits.
#
# ## Pipeline philosophy
#
# This notebook follows the OpenEnv hackathon best-practice recipe:
#   1. **Light SFT** to teach JSON action format without saturating entropy
#   2. **Memory-aware GRPO** with real env reward and only K=2 rollouts
#   3. **Eval sweep** showing baseline → SFT → GRPO progression
#
# Key design choices to prevent the two failure modes (overfit / underfit):
#   - **Diverse data**: 12 templates × 4 distinct evidence-gathering plans × 5
#     procgen variants = 240 expert episodes (~89% unique prompt-response pairs
#     after stratified sampling to 72 expert + 24 mediocre + 24 failure).
#   - **Short SFT**: keep the effective batch at 16 but lower the per-device
#     batch so the 7B model still fits cleanly with eval and checkpointing.
#   - **Short SFT instead of loss masking**: Qwen2.5's chat template lacks
#     `{% generation %}` markers, so we can't use `assistant_only_loss=True`.
#     50 steps × 0.8 epoch is short enough that even full-sequence loss won't
#     overfit on diverse data.
#   - **GRPO with 2 generations**: Unsloth's RL guidance requires at least two
#     samples for GRPO, and memory cost scales with the number of generations,
#     so we keep K=2 for the 7B run on one A100 80GB.
#
# ## How to run
#
# 1. **Run Cell 0 first** (idempotent — safe to re-run after kernel restart, ~5s).
# 2. Then run cells 1–13 top to bottom.
# 3. **After "Kernel → Restart"**: re-run Cell 0 + your target cell. Every cell
#    imports what it needs.
# 4. If a cell fails, read its printed **FIX** message.
#
# ## Resume points (skip done work)
# - `outputs/qwen25_7b_sft_final/` exists → Cell 6 skips and loads from disk.
# - `outputs/qwen25_7b_grpo_final/` exists → Cell 9 skips and loads from disk.
# - `eval/results/qwen25_7b_comparison_raw.csv` exists → Cell 10 skips and reuses results.
# - To force re-run, delete the corresponding artifact.

# %% [markdown]
# ## Cell 0 — Bootstrap (RUN ME FIRST after every kernel restart)
#
# Uses Unsloth's official `uv pip install` recipe with version pins that the
# Unsloth team has tested. Idempotent:
#   - First run on fresh Space: ~8-12 min (mostly downloading vLLM/torch)
#   - After kernel restart (deps still on disk): ~3-5 sec
#   - Re-run within same session: <1 sec
#
# **Pinned versions (DO NOT CHANGE without reading why):**
#   - `torch>=2.8.0` + `triton>=3.4.0` — Unsloth's tested CUDA stack
#   - `transformers==4.56.2` — required by Unsloth's chat template patches
#   - `trl==0.22.2` — has SFTConfig.assistant_only_loss + GRPOConfig.beta API
#   - `unsloth` from git — latest patches (the published wheel lags weeks)

# %%
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Unsloth's standby mode reduces RL memory pressure during vLLM-backed GRPO.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

GITHUB_USER = "Madhav-GPT"
REPO_NAME   = "sre-env"
BRANCH      = "main"

# ---- Step 1: ensure cwd is the repo root ----
if Path("sre_gym").exists() and Path("notebooks").exists():
    print(f"✓ Already in repo root: {Path('.').resolve()}")
elif Path(REPO_NAME).exists():
    os.chdir(REPO_NAME)
    print(f"✓ Changed to repo root: {Path('.').resolve()}")
else:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        url = f"https://{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
        print("Cloning with GITHUB_TOKEN from Space secret ...")
    else:
        url = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"
        print("Cloning public repo ...")
    subprocess.check_call(["git", "clone", "--depth=1", "--branch", BRANCH, url, REPO_NAME])
    os.chdir(REPO_NAME)
    print(f"✓ Cloned to: {Path('.').resolve()}")

REPO_ROOT = Path(".").resolve()
assert (REPO_ROOT / "sre_gym").exists(), "Wrong cwd — sre_gym/ not found"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Step 2: install deps (idempotent — checks first, installs only if missing) ----
def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None

REQUIRED = ["unsloth", "trl", "vllm", "datasets", "transformers",
            "matplotlib", "pandas", "httpx", "fastapi",
            "openenv",        # provided by openenv-core — used by unified_incident_env.client
            "uvicorn",        # used by openenv FastAPI server layer
            "websockets",     # used by openenv WS transport
            "yaml",           # provided by PyYAML — used by scenario config loaders
            ]

if all(_have(p) for p in REQUIRED) and _have("unified_incident_env"):
    print("✓ All deps already installed — skipping (~3s)")
else:
    print("Installing deps via Unsloth's uv pattern — first run is ~10 min on a fresh Space.")
    print("Progress prints below. If no new lines for >2 min, check `ps aux | grep uv` in Terminal.\n")

    # Step 2a: install uv (fast pip replacement Unsloth uses).
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-qqq", "uv"])

    # Match the user's existing numpy if any (avoids unnecessary reinstall).
    try:
        import numpy
        get_numpy = f"numpy=={numpy.__version__}"
    except ImportError:
        get_numpy = "numpy"

    # Step 2b: Unsloth's official core stack — single resolution pass with all
    # heavy packages so uv finds a coherent version set in one go.
    subprocess.check_call([
        "uv", "pip", "install", "-qqq", "--system",
        "torch>=2.8.0", "triton>=3.4.0", get_numpy, "torchvision", "bitsandbytes",
        "transformers==4.56.2", "trackio",
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
        "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        # Extras for our pipeline. Resolved together so vllm doesn't fight transformers.
        "vllm", "datasets", "accelerate", "matplotlib", "pandas",
        "httpx", "fastapi", "pydantic>=2.0",
        # Repo's own deps — without these, unified_incident_env.client fails:
        "openenv-core>=0.2.1",      # provides `openenv.core` namespace
        "uvicorn[standard]>=0.30.0",
        "websockets>=12.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ])

    # Step 2c: pin Unsloth's exact versions (--no-deps so we don't pull a
    # different transformers/tokenizers via dep resolution from another pkg).
    subprocess.check_call([
        "uv", "pip", "install", "--system", "--upgrade", "--no-deps",
        "transformers==4.56.2", "tokenizers", "trl==0.22.2", "unsloth", "unsloth_zoo",
    ])

    # Step 2d: install the repo itself without re-resolving deps.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps", "-e", ".",
    ])

    print("\n✓ All dependencies installed")

# ---- Step 3: GPU sanity check via nvidia-smi (BEFORE importing torch) ----
# nvidia-smi has clearer error messages than torch.cuda.is_available()
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
    capture_output=True, text=True,
)
if gpu.returncode != 0:
    raise RuntimeError(
        "\n  nvidia-smi failed — this Space has NO GPU.\n"
        "  FIX: Space → Settings → 'Space hardware' → 'Nvidia A100 Large 80GB' → Save\n"
        "  Then wait ~2 min for the Space to restart and re-run this cell.\n"
    )
print(f"\nGPU: {gpu.stdout.strip()}")
print(f"Repo root: {REPO_ROOT}")
print("\n✓ Cell 0 complete — proceed to Cell 1")

# %% [markdown]
# ## Cell 1 — Verify PyTorch sees the GPU
#
# If this cell prints a CUDA-mismatch error, follow the FIX instructions
# **exactly** — restart the kernel, re-run Cell 0, then re-run this cell.

# %%
import subprocess
import torch

if not torch.cuda.is_available():
    # Diagnose
    drv = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    raise RuntimeError(
        f"\n\n  PyTorch can't see the GPU.\n"
        f"  GPU driver: {drv}\n"
        f"  PyTorch built for CUDA: {torch.version.cuda}\n\n"
        f"  FIX: CUDA driver doesn't support PyTorch's CUDA build.\n"
        f"  This is rare on HF Spaces with A100 — usually means the Space\n"
        f"  hardware was changed mid-run. Try:\n"
        f"    1. Settings → Space hardware: confirm 'A100 Large 80GB'\n"
        f"    2. Restart Space (factory reset, not just kernel)\n"
        f"    3. Re-open notebook, run Cell 0 from scratch\n"
        f"  Do NOT downgrade torch — Unsloth requires torch>=2.8.0.\n"
    )

device_name = torch.cuda.get_device_name(0)
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f"GPU:        {device_name}")
print(f"VRAM:       {vram_gb} GB")
print(f"PyTorch:    {torch.__version__} (CUDA {torch.version.cuda})")

if vram_gb < 75:
    raise RuntimeError(
        f"\n  Need an A100 80GB-class GPU for Qwen2.5-7B + K=2 GRPO; got {vram_gb} GB.\n"
        f"  FIX: Space → Settings → upgrade to A100 80GB.\n"
    )
print("\n✓ GPU check passed")

# %% [markdown]
# ## Cell 2 — Build the SFT corpus
#
# Calls `train/build_corpus.py`. Idempotent: if `seed_v2_120.jsonl` already
# exists, only rebuilds if the upstream data changed.

# %%
import json
import subprocess
import sys
from pathlib import Path

CORPUS_PATH = Path("train/data/seed_v2_120.jsonl")

if CORPUS_PATH.exists():
    n = sum(1 for _ in open(CORPUS_PATH))
    print(f"Corpus exists at {CORPUS_PATH} ({n} episodes) — using as-is.")
    print("  Delete the file and re-run this cell to force rebuild.")
else:
    result = subprocess.run(
        [sys.executable, "train/build_corpus.py", "--output", str(CORPUS_PATH)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("build_corpus.py failed — see stderr above")

# %% [markdown]
# ## Cell 3 — Sanity-check the corpus
#
# 60/20/20 split, template-uniform coverage, score distribution. If any
# assertion fails, fix the corpus before training.

# %%
import json
from pathlib import Path

import pandas as pd

records = []
with open("train/data/seed_v2_120.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        records.append({
            "scenario_id": ep["scenario_id"],
            "template_id": ep["template_id"],
            "quality_tier": ep["quality_tier"],
            "final_score": ep["final_score"],
            "incident_resolved": ep["incident_resolved"],
            "steps": ep["steps"],
            "model": ep["model"],
        })
df = pd.DataFrame(records)

print(f"Episodes: {len(df)}\n")
print("Tier distribution:")
print(df.groupby("quality_tier").agg(
    n=("scenario_id", "count"),
    mean_score=("final_score", "mean"),
    min_score=("final_score", "min"),
    max_score=("final_score", "max"),
))
print("\nPer-template coverage:")
print(df["template_id"].value_counts().sort_index())

assert df["template_id"].nunique() == 12, "Missing template coverage"
assert df.groupby("template_id").size().min() >= 5, "Some templates have <5 episodes"
assert (df["quality_tier"] == "expert").sum() >= 60, "Expert tier too thin"
assert (df["quality_tier"] == "failure").sum() >= 20, "Failure tier too thin (GRPO will collapse)"
assert (df["final_score"] < 0.30).sum() >= 20, "Failure-band scores under-represented"
print("\n✓ Corpus sanity checks passed")

# %% [markdown]
# ## Cell 4 — Convert each step into a (prompt, completion) ChatML pair
#
# Drops steps with empty/short prompts, non-JSON responses, or >2048 tokens.

# %%
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

tokenizer_for_chat = AutoTokenizer.from_pretrained(MODEL_NAME)

SFT_SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no fences.
The JSON key for the action is "action_type" (NOT "action", NOT "type", NOT "tool").

Examples (copy this exact shape):
  {"action_type":"query_logs","service":"worker"}
  {"action_type":"query_metrics","service":"database","metric":"error_rate"}
  {"action_type":"run_check","check_name":"end_to_end"}
  {"action_type":"rollback_deploy","service":"worker"}
  {"action_type":"submit_hypothesis","hypothesis":{"root_cause":"bad_worker_deploy","affected_services":["worker"],"confidence":0.7,"recommended_next_action":"rollback_deploy"}}
  {"action_type":"declare_resolved"}

The 11 valid action_type values are:
  query_logs, query_metrics, query_dependencies, query_deploys,
  rollback_deploy, restart_service, isolate_service, run_check,
  submit_hypothesis, escalate, declare_resolved.

Services: api-gateway / cache / database / worker (no others).
metric in {cpu, error_rate, latency} (no others).
check_name in {database_recovery, end_to_end} (no others).

A successful episode looks like: gather evidence -> submit_hypothesis -> rollback ->
restart -> both run_checks pass -> declare_resolved. Wrong rollback / premature
restart / premature declare_resolved are penalized. Repeated identical hypotheses
score 0."""


def step_to_chatml(prompt, response):
    return tokenizer_for_chat.apply_chat_template(
        [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    )


pairs = []
with open("train/data/seed_v2_120.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        for step in ep["trajectory"]:
            prompt = step.get("prompt") or ""
            response = step.get("response_text") or ""
            if len(prompt) < 50:
                continue
            if not response.strip().startswith("{"):
                continue
            text = step_to_chatml(prompt, response)
            tokens = tokenizer_for_chat(text, return_length=True)["length"][0]
            if tokens > MAX_SEQ_LEN:
                continue
            pairs.append({"text": text, "tokens": tokens, "tier": ep["quality_tier"]})

print(f"SFT step-pairs: {len(pairs)}")
print("Token length distribution:")
print(pd.Series([p['tokens'] for p in pairs]).describe().round(0))

sft_dataset = Dataset.from_list([{"text": p["text"]} for p in pairs])
sft_dataset = sft_dataset.shuffle(seed=42)
print(f"\n✓ Built SFT dataset: {len(sft_dataset)} examples")

# %% [markdown]
# ## Cell 5 — Load Qwen2.5-7B-Instruct (4-bit + LoRA r=32)
#
# Keep the LoRA adapter small; the extra capacity comes from the larger base
# model, while the smaller adapter preserves headroom for GRPO on one A100.
# Skips download if already cached. ~3 min cold, <30 sec warm.

# %%
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                         # keep adapter modest; 7B capacity is in the base
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)
print("Trainable params:")
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 6 — SFT cold-start (50 steps) + perplexity gate
#
# This SFT is intentionally short. Goal: teach the JSON action format
# (Unsloth Advanced GRPO recipe calls this "prefinetuning to skip GRPO format
# learning"). We do NOT want the model to memorize specific action sequences
# — that would zero out GRPO's exploration variance.
#
# Anti-overfit measures vs typical SFT recipes:
#   - LoRA r=32, dropout=0.05 (small + lightly regularized — full 0.1 disables Unsloth's fast-patch path on 7B)
#   - lr=5e-5 (gentle), warmup 10%, cosine schedule
#   - 50 steps × batch 16 = 800 examples ≈ 0.8 epoch over our 999-step corpus
#   - full-sequence loss (Qwen2.5's chat template lacks generation markers
#     that would enable assistant_only_loss; the short SFT compensates)
#   - load_best_model_at_end + eval_loss → auto-restores the best checkpoint
#
# **Resume:** if `outputs/qwen25_7b_sft_final/` exists, skips training and loads from disk.
# To force re-train, delete the directory.

# %%
import math
from pathlib import Path

from trl import SFTTrainer, SFTConfig

SFT_OUT = Path("outputs/qwen25_7b_sft_final")

if SFT_OUT.exists() and (SFT_OUT / "adapter_model.safetensors").exists():
    print(f"✓ SFT already trained at {SFT_OUT} — loading adapter weights")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(SFT_OUT), is_trainable=True)
    print("Loaded SFT adapter — skipping SFT training")
    final_perplexity = None
else:
    train_eval = sft_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = train_eval["train"], train_eval["test"]
    print(f"SFT train: {len(train_ds)} | eval: {len(eval_ds)}")

    sft_args = SFTConfig(
        output_dir="outputs/qwen25_7b_sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=50,                  # was 150 — short SFT to avoid memorization
        learning_rate=5e-5,            # was 1e-4 — gentler LR for small data
        warmup_ratio=0.10,             # was 0.05 — slower start
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,                 # was 25 — catch overfitting earlier
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,   # auto-restore best eval-loss checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        # NOTE: assistant_only_loss=True needs the chat template to have
        # `{% generation %}` markers, which Qwen2.5-Instruct does NOT.
        # Enabling it would mask ALL tokens (silent broken training).
        # We rely on data diversity + dropout + short SFT instead.
        assistant_only_loss=False,
    )
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )
    print("\nStarting SFT cold-start ...")
    sft_trainer.train()
    sft_trainer.save_model(str(SFT_OUT))
    print(f"\n✓ Saved to {SFT_OUT}")

    eval_metrics = sft_trainer.evaluate()
    final_perplexity = math.exp(eval_metrics["eval_loss"])
    print(f"\nFinal eval perplexity (best checkpoint loaded): {final_perplexity:.3f}")

    if final_perplexity < 1.3:
        print(f"⚠ Perplexity {final_perplexity:.3f} < 1.3 — policy may be too deterministic.")
        print("  Watch GRPO reward variance in first 10 steps.")
        print("  If std≈0 across K=2 rollouts: abort GRPO, ship SFT-only.")
    elif final_perplexity > 5.0:
        raise RuntimeError(
            f"\n  Perplexity {final_perplexity:.3f} > 5.0 — SFT undercooked.\n"
            f"  FIX: Delete outputs/qwen25_7b_sft_final/, bump max_steps to 100, re-run Cell 6."
        )
    elif 1.5 <= final_perplexity <= 3.0:
        print("✓ Perplexity in healthy band [1.5, 3.0] — proceed to GRPO")
    else:
        print(f"  Perplexity {final_perplexity:.3f} acceptable [1.3, 5.0] — proceed to GRPO")

# %% [markdown]
# ## Cell 7 — Build the GRPO prompt dataset
#
# Sample observations from the env after a varying number of warmup steps so
# the model sees prompts at different workflow stages. Each prompt carries
# `scenario_id` so the reward function can replay the env to the same state.

# %%
import random
import sys
from pathlib import Path

from datasets import Dataset

if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.environment import UnifiedIncidentEnvironment

# Inline the SFT system prompt so this cell survives kernel restart standalone
# (must be IDENTICAL to the one in Cell 4 — same chat template = same model
# distribution at test time).
SFT_SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no fences.
The JSON key for the action is "action_type" (NOT "action", NOT "type", NOT "tool").

Examples (copy this exact shape):
  {"action_type":"query_logs","service":"worker"}
  {"action_type":"query_metrics","service":"database","metric":"error_rate"}
  {"action_type":"run_check","check_name":"end_to_end"}
  {"action_type":"rollback_deploy","service":"worker"}
  {"action_type":"submit_hypothesis","hypothesis":{"root_cause":"bad_worker_deploy","affected_services":["worker"],"confidence":0.7,"recommended_next_action":"rollback_deploy"}}
  {"action_type":"declare_resolved"}

The 11 valid action_type values are:
  query_logs, query_metrics, query_dependencies, query_deploys,
  rollback_deploy, restart_service, isolate_service, run_check,
  submit_hypothesis, escalate, declare_resolved.

Services: api-gateway / cache / database / worker (no others).
metric in {cpu, error_rate, latency} (no others).
check_name in {database_recovery, end_to_end} (no others).

A successful episode looks like: gather evidence -> submit_hypothesis -> rollback ->
restart -> both run_checks pass -> declare_resolved. Wrong rollback / premature
restart / premature declare_resolved are penalized. Repeated identical hypotheses
score 0."""


def build_grpo_prompts(num_prompts=120, seed=0):
    """Build GRPO prompts already wrapped in the SAME chat template that SFT used.

    CRITICAL: TRL's GRPOTrainer treats the dataset's "prompt" column as raw
    text and feeds it straight to model.generate. If we pass raw env prompts
    (no <|im_start|> markers), the SFT-trained model sees out-of-distribution
    input, doesn't produce JSON, and every rollout gets the -0.5 format
    penalty (reward_std=0 → no learning signal).

    The fix: pre-render the chat template here so the prompt looks identical
    to what the model saw during SFT, ending with `<|im_start|>assistant\n`.
    """
    templates = [
        "worker_deploy_cascade", "db_config_rollout", "gateway_auth_rollout",
        "payment_webhook_misconfig", "schema_drift_missing_migration", "cache_stale_state",
        "dep_degradation", "memory_leak_oom", "auth_token_expiry",
        "network_partition", "rate_limit_retry_storm", "migration_lock",
    ]
    rng = random.Random(seed)
    out = []
    for i in range(num_prompts):
        base = templates[i % len(templates)]
        scenario = base if i < len(templates) else f"{base}__p0{1 + (i // len(templates)) % 4}"
        env = UnifiedIncidentEnvironment()
        try:
            obs = env.reset(scenario_id=scenario)
        except Exception:
            scenario = base
            obs = env.reset(scenario_id=scenario)
        if rng.random() > 0.5:
            obs = env.step(UnifiedIncidentAction(action_type="query_logs", service="worker"))

        # Render the prompt in the same ChatML format SFT trained with.
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": obs.prompt_text},
            ],
            tokenize=False,
            add_generation_prompt=True,   # appends `<|im_start|>assistant\n`
        )
        out.append({"prompt": chat_prompt, "scenario_id": scenario})
    return out


grpo_prompts_list = build_grpo_prompts(num_prompts=120, seed=42)
grpo_prompts_ds = Dataset.from_list(grpo_prompts_list)
print(f"✓ Built {len(grpo_prompts_ds)} GRPO prompts")
print(f"Sample: {grpo_prompts_ds[0]['scenario_id']}")

# %% [markdown]
# ## Cell 8 — Per-turn proxy reward function
#
# For each completion (one JSON action):
# 1. Parse JSON (-0.5 if invalid)
# 2. Validate schema (-0.3 if invalid)
# 3. Reset env to scenario, step once
# 4. Return env reward + bonuses

# %%
import json
import sys
from pathlib import Path

if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.environment import UnifiedIncidentEnvironment

# Action types and their required fields — used by the partial-credit ladder.
_VALID_ACTION_TYPES = {
    "query_logs", "query_metrics", "query_dependencies", "query_deploys",
    "rollback_deploy", "restart_service", "isolate_service",
    "run_check", "submit_hypothesis", "escalate", "declare_resolved",
}
_QUERY_ACTIONS = {"query_logs", "query_metrics", "query_dependencies", "query_deploys"}
_REQUIRES_SERVICE = {
    "query_logs", "query_dependencies", "query_deploys",
    "rollback_deploy", "restart_service", "isolate_service",
}
_VALID_SERVICES = {"api-gateway", "cache", "database", "worker"}
_VALID_METRICS = {"cpu", "error_rate", "latency"}
_VALID_CHECKS = {"database_recovery", "end_to_end"}

# LLM-friendly aliases the SFT'd model hallucinates instead of exact schema names.
_KEY_ALIASES = {
    "action": "action_type", "actiontype": "action_type", "action_name": "action_type",
    "type": "action_type", "tool": "action_type", "tool_name": "action_type",
    "command": "action_type", "service_name": "service", "target": "service",
    "metric_name": "metric", "check": "check_name", "check_id": "check_name",
}

_DEBUG_PRINTED = {"n": 0}


def _extract_action_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.startswith("```")).strip()
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None


def _normalize_action_dict(d):
    """Remap common LLM key hallucinations to schema field names."""
    if not isinstance(d, dict):
        return d, 0
    out = {}
    n_corrections = 0
    for k, v in d.items():
        if k in _KEY_ALIASES:
            mapped = _KEY_ALIASES[k]
            if mapped not in d and mapped not in out:
                out[mapped] = v
                n_corrections += 1
                continue
            n_corrections += 1
            continue
        out[k] = v
    return out, n_corrections


def _grade_partial(action_dict):
    """Partial-credit ladder for actions that fail strict schema validation."""
    if not isinstance(action_dict, dict):
        return -0.45
    if "action_type" not in action_dict:
        return -0.40
    at = action_dict["action_type"]
    if not isinstance(at, str):
        return -0.35
    if at not in _VALID_ACTION_TYPES:
        return -0.30
    if at in _REQUIRES_SERVICE and not action_dict.get("service"):
        return -0.25
    if at == "query_metrics" and (not action_dict.get("service") or not action_dict.get("metric")):
        return -0.25
    if at == "run_check" and not action_dict.get("check_name"):
        return -0.25
    if at == "submit_hypothesis" and not isinstance(action_dict.get("hypothesis"), dict):
        return -0.25
    if "service" in action_dict and action_dict["service"] not in _VALID_SERVICES:
        return -0.20
    if "metric" in action_dict and action_dict.get("metric") not in _VALID_METRICS:
        return -0.15
    if "check_name" in action_dict and action_dict.get("check_name") not in _VALID_CHECKS:
        return -0.15
    if at == "submit_hypothesis":
        h = action_dict.get("hypothesis", {})
        required = {"root_cause", "affected_services", "confidence", "recommended_next_action"}
        if not required.issubset(h.keys()):
            return -0.10
    return -0.05


def _first_action_bonus(action, obs, truth, recipe):
    """Scenario-aware bonus for the model's first action.

    The env's per-step shaping reward is small (~0.01 in absolute value), so
    rollouts collapse to similar rewards even when they pick different
    actions. This bonus reads truth.best_next_action and
    remediation_recipe.rollback_target from the scenario and rewards the
    model for matching expert behavior.

    Returns a float in [-0.40, +0.50].
    """
    bonus = 0.0
    at = action.action_type
    affected = set(truth.get("affected_services") or [])
    best = truth.get("best_next_action") or ""
    true_root = truth.get("root_cause")
    rollback_target = recipe.get("rollback_target")

    # Strong rollback signal — the single most important first move
    if at == "rollback_deploy":
        if action.service == rollback_target:
            bonus += 0.40
        else:
            bonus -= 0.30

    # Strong hypothesis signal — second most important
    if at == "submit_hypothesis" and action.hypothesis is not None:
        h_root = getattr(action.hypothesis, "root_cause", None)
        if h_root == true_root:
            bonus += 0.35
        else:
            bonus -= 0.10

    # Match to env's recommended next action
    if at == best and at not in {"rollback_deploy", "submit_hypothesis"}:
        bonus += 0.30

    # Query the affected service vs a noise/non-affected service
    if at in _QUERY_ACTIONS:
        if action.service in affected:
            bonus += 0.15
        elif action.service in _VALID_SERVICES:
            bonus -= 0.05  # querying a real service that's not actually affected

    # declare_resolved — only valid if env confirms
    if at == "declare_resolved":
        if obs.incident_resolved:
            bonus += 0.50
        else:
            bonus -= 0.30

    # Escalation = giving up; small penalty
    if at == "escalate":
        bonus -= 0.10

    return bonus


def reward_fn(completions, prompts=None, **kwargs):
    """Per-turn proxy reward with scenario-aware first-action bonuses.

    Composite score:
      r = env_obs.reward                       # env shaping (~-0.02..+0.05)
        + _first_action_bonus(...)             # scenario-aware (-0.40..+0.50)
        + 0.50 if incident_resolved
        - 0.20 if obs.failure_type
        - 0.05 * n_aliases_used
    """
    scenario_ids = kwargs.get("scenario_id") or [None] * len(completions)
    rewards = []
    debug_samples = []

    for completion, scenario_id in zip(completions, scenario_ids):
        if scenario_id is None:
            rewards.append(0.0)
            continue
        action_dict = _extract_action_json(completion)
        if action_dict is None:
            rewards.append(-0.5)
            if _DEBUG_PRINTED["n"] < 3:
                debug_samples.append((completion, action_dict, "json_extract_failed", -0.5, 0, 0.0))
            continue
        normalized, n_aliases = _normalize_action_dict(action_dict)
        try:
            action = UnifiedIncidentAction(**normalized)
        except Exception as exc:
            r = _grade_partial(normalized)
            rewards.append(r)
            if _DEBUG_PRINTED["n"] < 3:
                debug_samples.append((completion, normalized, f"schema_fail: {exc}", r, n_aliases, 0.0))
            continue

        env = UnifiedIncidentEnvironment()
        try:
            env.reset(scenario_id=scenario_id)
        except Exception as exc:
            rewards.append(-0.2)
            if _DEBUG_PRINTED["n"] < 3:
                debug_samples.append((completion, normalized, f"env_reset_fail: {exc}", -0.2, n_aliases, 0.0))
            continue

        # Read truth BEFORE stepping (env mutates _episode each step)
        scenario_dict = (env._episode or {}).get("scenario", {})
        truth = scenario_dict.get("truth", {}) or {}
        recipe = scenario_dict.get("remediation_recipe", {}) or {}

        try:
            obs = env.step(action)
        except Exception as exc:
            rewards.append(-0.2)
            if _DEBUG_PRINTED["n"] < 3:
                debug_samples.append((completion, normalized, f"env_step_fail: {exc}", -0.2, n_aliases, 0.0))
            continue

        env_r = float(obs.reward)
        bonus = _first_action_bonus(action, obs, truth, recipe)

        r = env_r + bonus
        if obs.failure_type:
            r -= 0.20
        if obs.incident_resolved:
            r += 0.50
        r -= 0.05 * n_aliases
        rewards.append(r)

        if _DEBUG_PRINTED["n"] < 3:
            tag = f"env_ok r={env_r:+.3f} bonus={bonus:+.3f}"
            debug_samples.append((completion, normalized, tag, r, n_aliases, bonus))

    if _DEBUG_PRINTED["n"] < 3 and debug_samples:
        print("\n[reward_fn debug] First-call sample completions:")
        for i, (comp, parsed, reason, r, n_a, bonus) in enumerate(debug_samples[:3]):
            print(f"  ── sample {i} (reward={r:+.3f}, n_aliases={n_a}, {reason}) ──")
            preview = comp[:300].replace("\n", " | ")
            print(f"     completion: {preview!r}")
            if parsed is not None:
                print(f"     normalized: {parsed}")
        _DEBUG_PRINTED["n"] = 3
        print("[reward_fn debug] Reward composition:")
        print("  r = env.reward + scenario_aware_bonus + resolved_bonus - failure_penalty - 0.05·n_aliases")
        print("  Bonus rubric — pushes the model toward scenario-specific expert behavior:")
        print("    +0.40 correct rollback / +0.35 correct hypothesis / +0.30 best_next_action match")
        print("    +0.15 querying an affected service / +0.50 declare_resolved when actually resolved")
        print("    -0.30 wrong rollback / premature declare_resolved")
        print("    -0.10 wrong hypothesis / escalate / -0.05 querying a non-affected service\n")

    return rewards


# Smoke tests
print("[smoke] reward_fn validation:")
_t1 = reward_fn(['{"action_type":"query_deploys","service":"worker"}'],
                scenario_id=["worker_deploy_cascade"])
print(f"  query_deploys worker (right service):     {_t1[0]:+.3f}  (expect: positive — affected service bonus)")

_t2 = reward_fn(['{"action_type":"query_deploys","service":"cache"}'],
                scenario_id=["worker_deploy_cascade"])
print(f"  query_deploys cache  (wrong service):     {_t2[0]:+.3f}  (expect: negative — non-affected penalty)")

_t3 = reward_fn(['{"action_type":"rollback_deploy","service":"worker"}'],
                scenario_id=["worker_deploy_cascade"])
print(f"  rollback_deploy worker (correct target):  {_t3[0]:+.3f}  (expect: large positive)")

_t4 = reward_fn(['{"action_type":"rollback_deploy","service":"cache"}'],
                scenario_id=["worker_deploy_cascade"])
print(f"  rollback_deploy cache  (wrong target):    {_t4[0]:+.3f}  (expect: large negative)")

_t5 = reward_fn(['{"action_type":"declare_resolved"}'],
                scenario_id=["worker_deploy_cascade"])
print(f"  declare_resolved (premature):             {_t5[0]:+.3f}  (expect: negative)")

# Validate spread — these 5 should be 5 distinct values for GRPO variance
unique = sorted({round(t[0], 2) for t in [_t1, _t2, _t3, _t4, _t5]})
print(f"  Distinct reward levels across 5 actions:  {len(unique)} ({unique})")
assert len(unique) >= 4, "Reward function must give clearly distinct values for K=2 GRPO variance"

# Reset debug counter so the first real GRPO call still gets to print
_DEBUG_PRINTED["n"] = 0
print("✓ reward_fn validated (scenario-aware first-action bonuses + alias normalization + partial ladder)")

# %% [markdown]
# ## Cell 9 — GRPO online training (40 steps × K=2)
#
# Wall-clock: ~30-45 min on A100 (vLLM path), ~90 min on transformers fallback.
# Watch:
#   - **reward mean** should rise from ~0 → ~0.4 by step 30
#   - **reward std** must be > 0.05 in steps 1-5 (else K=2 rollouts are
#     identical and GRPO has no advantage signal — abort and ship SFT-only)
#   - if reward flatlines after step 25, interrupt — no more signal
#
# Anti-collapse settings vs default GRPO:
#   - beta=0.1 (was 0.04) — stronger KL penalty keeps policy near SFT
#   - temperature=0.9 (was 0.7) — more rollout variance
#   - lr=1e-6 (was 5e-6) — gentler updates
#   - top_p=0.95 — nucleus sampling for stability
#
# **vLLM rollouts** are auto-detected:
#   - If Cell 5 loaded the model with `fast_inference=True`, the model exposes
#     `model.vllm_engine` and we use vLLM (3× faster).
#   - Otherwise we fall back to transformers `model.generate` — slower but
#     always works. **No retry loop** — a single attempt with the right setting
#     avoids the `EADDRINUSE` collision on `MASTER_PORT` that the old try/except
#     fallback caused after a failed init.
#
# **Resume:** if `outputs/qwen25_7b_grpo_final/` exists, skips training.
#
# **If you hit `EADDRINUSE` on MASTER_PORT after a previous broken run:**
# restart the kernel, re-run Cell 0 + Cells 1–8, then re-run this cell. The
# defensive `destroy_process_group()` below clears any in-process state, but a
# port left bound by a sub-process from a prior crash needs a kernel restart.

# %%
import os
from pathlib import Path

import torch.distributed as dist
from trl import GRPOConfig, GRPOTrainer

GRPO_OUT = Path("outputs/qwen25_7b_grpo_final")


def _model_has_vllm_engine(m) -> bool:
    """Return True iff the model exposes a working `vllm_engine` attribute.

    Unsloth attaches `vllm_engine` only when `FastLanguageModel.from_pretrained`
    was called with `fast_inference=True`. PEFT's `__getattr__` proxies through
    the wrapper chain (PeftModelForCausalLM -> LoraModel -> base model) and
    raises AttributeError at the bottom if no member supplies it. Probing via
    try/except is the only safe way — `hasattr` swallows non-AttributeError
    side effects in some torch builds.
    """
    try:
        engine = m.vllm_engine
    except AttributeError:
        return False
    return engine is not None


def _cleanup_distributed():
    """Drop any stale torch.distributed process group from a prior failed init.

    The original Cell 9 wrapped GRPOTrainer in try/except and retried with
    use_vllm=False on failure. But GRPOConfig.__post_init__ runs
    `init_process_group` on the same MASTER_PORT, so the retry tripped
    EADDRINUSE. We avoid retries entirely now, but we still call destroy
    defensively in case the user re-runs this cell after a previous crash.
    """
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _build_grpo_args(use_vllm: bool):
    # Tuned for our small/uniform-ish dataset to maintain exploration:
    #   - higher beta (KL penalty) keeps policy near SFT, prevents drift
    #   - higher temperature in rollouts → more variance between K=2 generations
    #     (without it: std≈0, no advantage signal, no learning)
    #   - lower LR is gentler since per-step KL penalty already constrains updates
    return GRPOConfig(
        output_dir="outputs/qwen25_7b_grpo",
        num_generations=2,
        max_steps=40,
        learning_rate=1e-6,            # was 5e-6 — more stable
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_prompt_length=1536,
        max_completion_length=192,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=0.35 if use_vllm else 0.0,
        beta=0.1,                      # was 0.04 — stronger KL anchor
        temperature=0.9,               # was 0.7 — more rollout variance
        top_p=0.95,
        logging_steps=2,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        report_to="none",
    )


if GRPO_OUT.exists() and (GRPO_OUT / "adapter_model.safetensors").exists():
    print(f"✓ GRPO already trained at {GRPO_OUT} — loading adapter weights")
    if hasattr(model, "load_adapter"):
        model.load_adapter(str(GRPO_OUT), adapter_name="default")
    print("Loaded GRPO adapter — skipping GRPO training")
else:
    # Pre-flight: vLLM rollouts only work if Cell 5 loaded the model with
    # `fast_inference=True`. Detect once instead of try/except retry — the
    # retry leaves a half-init process group that breaks the next attempt.
    use_vllm = _model_has_vllm_engine(model)
    if use_vllm:
        print("✓ Detected `model.vllm_engine` — using vLLM rollouts (fast path)")
    else:
        print(
            "ℹ vLLM engine not attached to model "
            "(Cell 5 didn't set `fast_inference=True`). "
            "Using transformers `model.generate` for GRPO rollouts — "
            "~3× slower but always works. To enable vLLM next run, "
            "re-load the model in Cell 5 with `fast_inference=True, "
            "max_lora_rank=32, gpu_memory_utilization=0.35`."
        )

    # Pin a deterministic MASTER_PORT/MASTER_ADDR so a botched init from a
    # prior cell can't collide and trigger EADDRINUSE.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    _cleanup_distributed()

    print(f"Starting GRPO online training (use_vllm={use_vllm}) ...")
    grpo_trainer = GRPOTrainer(
        model=model,
        args=_build_grpo_args(use_vllm=use_vllm),
        reward_funcs=[reward_fn],
        train_dataset=grpo_prompts_ds,
        processing_class=tokenizer,
    )
    grpo_trainer.train()

    grpo_trainer.save_model(str(GRPO_OUT))
    print(f"✓ Saved to {GRPO_OUT}")

# %% [markdown]
# ## Cell 10 — Eval comparison sweep
#
# Up to 5 policies × 12 holdout × 3 seeds. Saves to
# `eval/results/qwen25_7b_comparison_raw.csv`. Skips if results already exist.
#
# **Robust to missing artifacts:**
#   - If `outputs/qwen25_7b_grpo_final/` doesn't exist (user skipped GRPO due to
#     collapse), the GRPO row is omitted automatically.
#   - If loading SFT-only as a second model fails (OOM), the SFT-only row is
#     omitted and we still report random/heuristic/scripted/grpo.

# %%
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from unsloth import FastLanguageModel

if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.environment import UnifiedIncidentEnvironment
from unified_incident_env.server.challenge import list_baselines

EVAL_CSV = Path("eval/results/qwen25_7b_comparison_raw.csv")
MAX_SEQ_LEN = 2048

# System prompt — duplicated from Cell 4 so this cell is self-contained
# (survives kernel restart if user re-runs only this cell after Cell 0).
SFT_SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no fences.
The JSON key for the action is "action_type" (NOT "action", NOT "type", NOT "tool").

Examples (copy this exact shape):
  {"action_type":"query_logs","service":"worker"}
  {"action_type":"query_metrics","service":"database","metric":"error_rate"}
  {"action_type":"run_check","check_name":"end_to_end"}
  {"action_type":"rollback_deploy","service":"worker"}
  {"action_type":"submit_hypothesis","hypothesis":{"root_cause":"bad_worker_deploy","affected_services":["worker"],"confidence":0.7,"recommended_next_action":"rollback_deploy"}}
  {"action_type":"declare_resolved"}

The 11 valid action_type values are:
  query_logs, query_metrics, query_dependencies, query_deploys,
  rollback_deploy, restart_service, isolate_service, run_check,
  submit_hypothesis, escalate, declare_resolved.

Services: api-gateway / cache / database / worker (no others).
metric in {cpu, error_rate, latency} (no others).
check_name in {database_recovery, end_to_end} (no others).

A successful episode looks like: gather evidence -> submit_hypothesis -> rollback ->
restart -> both run_checks pass -> declare_resolved. Wrong rollback / premature
restart / premature declare_resolved are penalized. Repeated identical hypotheses
score 0."""

if EVAL_CSV.exists():
    results_df = pd.read_csv(EVAL_CSV)
    print(f"✓ Eval already done — loaded {len(results_df)} rows from {EVAL_CSV}")
else:
    holdout = json.load(open("eval/holdout_basic.json"))
    HOLDOUT_SCENARIOS = holdout["scenario_ids"]
    print(f"Holdout: {len(HOLDOUT_SCENARIOS)} scenarios")

    # Detect what models we have
    has_grpo = Path("outputs/qwen25_7b_grpo_final/adapter_model.safetensors").exists()
    has_sft  = Path("outputs/qwen25_7b_sft_final/adapter_model.safetensors").exists()
    print(f"SFT artifact: {'✓' if has_sft else '✗'}   GRPO artifact: {'✓' if has_grpo else '✗'}")
    ACTIVE_ADAPTER_DIR = Path("outputs/qwen25_7b_grpo_final") if has_grpo else Path("outputs/qwen25_7b_sft_final")

    def load_local_adapter(adapter_dir: Path):
        from peft import PeftModel

        loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
            dtype=None,
        )
        loaded_model = PeftModel.from_pretrained(
            loaded_model,
            str(adapter_dir),
            is_trainable=False,
        )
        FastLanguageModel.for_inference(loaded_model)
        return loaded_model, loaded_tokenizer

    if "model" in globals() and "tokenizer" in globals():
        active_model, active_tokenizer = model, tokenizer
        FastLanguageModel.for_inference(active_model)
        print("✓ Using in-memory model/tokenizer from earlier notebook cells")
    elif ACTIVE_ADAPTER_DIR.exists() and (ACTIVE_ADAPTER_DIR / "adapter_model.safetensors").exists():
        active_model, active_tokenizer = load_local_adapter(ACTIVE_ADAPTER_DIR)
        print(f"✓ Loaded active adapter from {ACTIVE_ADAPTER_DIR}")
    else:
        raise RuntimeError(
            "\n  No local adapter was found for eval.\n"
            "  FIX: run Cell 6 (SFT) or Cell 9 (GRPO) first so outputs exist on disk.\n"
        )

    # Try to load SFT-only as a second model for comparison.
    # If it OOMs or any other error, we skip the SFT-only row.
    sft_only_model = None
    if has_sft and has_grpo:
        try:
            sft_only_model, _ = load_local_adapter(Path("outputs/qwen25_7b_sft_final"))
            print("✓ Loaded SFT-only adapter for comparison")
        except Exception as exc:
            # Catches OOM, ImportError (peft/transformers mismatch on tensor-parallel
            # imports), AttributeError (Unsloth + adapter mismatch), etc. Eval still
            # runs with 4 policies (random, heuristic, scripted, GRPO) instead of 5.
            print(f"⚠ Could not load SFT-only model ({type(exc).__name__}: {exc})")
            print("  Skipping SFT-vs-GRPO comparison row; eval proceeds with 4 policies.")
            sft_only_model = None
            torch.cuda.empty_cache()

    def _extract_json(text):
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
        s = text.find("{"); e = text.rfind("}")
        if s < 0 or e <= s: return None
        try: return json.loads(text[s:e+1])
        except: return None

    def lm_action(prompt, lm, max_new=120):
        ids = active_tokenizer.apply_chat_template(
            [{"role": "system", "content": SFT_SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(lm.device)
        out = lm.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=active_tokenizer.eos_token_id)
        text = active_tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return _extract_json(text) or {"action_type": "escalate"}

    def run_lm(scenario_id, seed, lm, max_steps=15):
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=scenario_id, seed=seed)
        for _ in range(max_steps):
            d = lm_action(obs.prompt_text or "", lm)
            try: action = UnifiedIncidentAction(**d)
            except: action = UnifiedIncidentAction(action_type="escalate")
            obs = env.step(action)
            if obs.done: break
        return obs

    def run_callable(scenario_id, seed, policy, max_steps=15):
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=scenario_id, seed=seed)
        for _ in range(max_steps):
            obs = env.step(policy(env, obs))
            if obs.done: break
        return obs

    def random_policy(env, obs):
        rng = random.Random(env._episode["tick"] + hash(obs.prompt_text) % 1000)
        atype = rng.choice([
            "query_logs","query_metrics","query_dependencies","query_deploys",
            "rollback_deploy","restart_service","isolate_service","run_check",
            "declare_resolved","escalate",
        ])
        kw = {"action_type": atype}
        if atype in {"query_logs","query_dependencies","query_deploys",
                     "rollback_deploy","restart_service","isolate_service"}:
            kw["service"] = rng.choice(["api-gateway","cache","database","worker"])
        if atype == "query_metrics":
            kw["service"] = rng.choice(["api-gateway","cache","database","worker"])
            kw["metric"] = rng.choice(["cpu","error_rate","latency"])
        if atype == "run_check":
            kw["check_name"] = rng.choice(["database_recovery","end_to_end"])
        try: return UnifiedIncidentAction(**kw)
        except: return UnifiedIncidentAction(action_type="escalate")

    def heuristic_policy(env, obs):
        truth = env._episode["scenario"]["truth"]
        tick = env._episode["tick"]
        if tick == 0: return UnifiedIncidentAction(action_type="query_logs", service="worker")
        if tick == 1: return UnifiedIncidentAction(action_type="query_deploys", service="worker")
        if tick == 2:
            affected = list(truth.get("affected_services") or [])[:1] or ["worker"]
            return UnifiedIncidentAction(
                action_type="submit_hypothesis",
                hypothesis={
                    "root_cause": truth["root_cause"],
                    "affected_services": affected,
                    "confidence": 0.7,
                    "recommended_next_action": truth.get("best_next_action") or "rollback_deploy",
                },
            )
        return UnifiedIncidentAction(action_type="escalate")

    def scripted_for(scenario_id):
        actions = [s.action for s in list_baselines(scenario_id=scenario_id).baselines[0].actions]
        cur = {"i": 0}
        def policy(env, obs):
            if cur["i"] >= len(actions):
                return UnifiedIncidentAction(action_type="escalate")
            a = actions[cur["i"]]; cur["i"] += 1
            return a
        return policy

    def _row(policy, sid, seed, obs):
        return {"policy": policy, "scenario_id": sid, "seed": seed,
                "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                "steps": obs.tick_count}

    # Determine final model label based on what was trained
    final_model_label = "qwen25-7b-grpo" if has_grpo else "qwen25-7b-sft"

    results = []
    for sid in HOLDOUT_SCENARIOS:
        print(f"  {sid} ...")
        for seed in range(3):
            results.append(_row("random",            sid, seed, run_callable(sid, seed, random_policy)))
            results.append(_row("heuristic",         sid, seed, run_callable(sid, seed, heuristic_policy)))
            results.append(_row("scripted_optimal", sid, seed, run_callable(sid, seed, scripted_for(sid))))
            if sft_only_model is not None:
                results.append(_row("qwen25-7b-sft-only", sid, seed, run_lm(sid, seed, sft_only_model)))
            results.append(_row(final_model_label,    sid, seed, run_lm(sid, seed, active_model)))

    results_df = pd.DataFrame(results)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(EVAL_CSV, index=False)
    print(f"\n✓ Saved {len(results_df)} eval rows to {EVAL_CSV}")

# %% [markdown]
# ## Cell 11 — Summary table + hero plot

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results_df = pd.read_csv("eval/results/qwen25_7b_comparison_raw.csv")
holdout = json.load(open("eval/holdout_basic.json"))
HOLDOUT_SCENARIOS = holdout["scenario_ids"]

summary = results_df.groupby("policy").agg(
    mean=("final_score", "mean"),
    median=("final_score", "median"),
    p25=("final_score", lambda x: x.quantile(0.25)),
    p75=("final_score", lambda x: x.quantile(0.75)),
    resolved_rate=("incident_resolved", "mean"),
).round(3).sort_values("mean")
summary.to_csv("eval/results/qwen25_7b_comparison_summary.csv")
print(summary)

fig, ax = plt.subplots(figsize=(9, 5))
order = summary.index.tolist()
# Clip yerr to >= 0: when a policy has tail outliers the mean can fall
# below the 25th percentile, producing a negative `mean - p25` that
# matplotlib rejects. Clipping collapses the impossible side to zero.
yerr_lo = (summary["mean"] - summary["p25"]).clip(lower=0)
yerr_hi = (summary["p75"] - summary["mean"]).clip(lower=0)
ax.bar(order, summary["mean"],
       yerr=[yerr_lo, yerr_hi],
       capsize=5, color="#3a86ff")
ax.axhline(0.65, ls="--", color="gray", alpha=0.5, label="heuristic floor (0.65)")
ax.axhline(0.80, ls="--", color="gray", alpha=0.5, label="heuristic ceiling (0.80)")
ax.axhline(0.90, ls="--", color="green", alpha=0.5, label="scripted reference (0.90)")
ax.set_ylabel("Final score (5-component composite)")
ax.set_xlabel("Policy")
ax.set_title("sre-gym Triage holdout eval (Qwen2.5-7B, 12 scenarios × 3 seeds)")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", fontsize=8)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("eval/results/qwen25_7b_comparison_hero.png", dpi=150)
plt.show()

# Per-template plot — uses only the policies actually present in the results
fig, ax = plt.subplots(figsize=(13, 6))
template_ids = sorted({s.split("__")[0] for s in HOLDOUT_SCENARIOS})
positions = list(range(len(template_ids)))
policies_present = list(summary.index)        # dynamic — survives missing rows
bar_width = 0.8 / max(len(policies_present), 1)
for offset, policy in enumerate(policies_present):
    sub = results_df[results_df.policy == policy]
    means = [sub[sub.scenario_id.str.startswith(t)]["final_score"].mean() for t in template_ids]
    ax.bar([p + offset * bar_width for p in positions], means, width=bar_width, label=policy)
ax.set_xticks([p + bar_width * len(policies_present) / 2 for p in positions])
ax.set_xticklabels(template_ids, rotation=30, ha="right")
ax.set_ylabel("Mean final_score")
ax.legend(fontsize=8)
ax.set_title("Per-template mean score by policy")
plt.tight_layout()
plt.savefig("eval/results/qwen25_7b_comparison_per_template.png", dpi=150)
plt.show()

# %% [markdown]
# ## Cell 12 — Push the final adapter to Hugging Face Hub
#
# Paste your write-enabled `HF_TOKEN` manually below. This cell prefers the
# trainer-native `push_to_hub()` path when a trainer object is still in memory,
# and otherwise falls back to reloading the saved adapter from disk.

# %%
from pathlib import Path

from unsloth import FastLanguageModel

HF_TOKEN = ""
HUB_MODEL_ID = "your-username/sre-gym-triage-qwen25-7b-grpo"
COMMIT_MESSAGE = "Qwen2.5-7B GRPO trained on sre-gym Triage"
MAX_SEQ_LEN = 2048

PUSH_SOURCE = Path("outputs/qwen25_7b_grpo_final")
if not (PUSH_SOURCE / "adapter_model.safetensors").exists():
    PUSH_SOURCE = Path("outputs/qwen25_7b_sft_final")

if not HF_TOKEN.strip():
    raise ValueError("Paste your HF token into HF_TOKEN before running this cell.")
if not (PUSH_SOURCE / "adapter_model.safetensors").exists():
    raise FileNotFoundError(
        "No saved adapter found. Run Cell 6 (SFT) or Cell 9 (GRPO) before pushing."
    )

print(f"Pushing to: {HUB_MODEL_ID}")
push_ok = False
trainer_for_push = globals().get("grpo_trainer") or globals().get("sft_trainer")

if trainer_for_push is not None:
    try:
        trainer_for_push.push_to_hub(
            repo_id=HUB_MODEL_ID,
            token=HF_TOKEN,
            commit_message=COMMIT_MESSAGE,
        )
        print(f"Pushed via trainer: https://huggingface.co/{HUB_MODEL_ID}")
        push_ok = True
    except Exception as exc:
        print(f"Trainer push failed: {exc}")
        print(f"Falling back to adapter push from: {PUSH_SOURCE}")

if not push_ok:
    from peft import PeftModel

    model_for_push, tokenizer_for_push = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model_for_push = PeftModel.from_pretrained(
        model_for_push,
        str(PUSH_SOURCE),
        is_trainable=False,
    )
    model_for_push.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN, commit_message=COMMIT_MESSAGE)
    tokenizer_for_push.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"Pushed via adapter fallback: https://huggingface.co/{HUB_MODEL_ID}")

# %% [markdown]
# ## Cell 13 — Package artifacts for download
#
# Tars the final artifacts so you can download a single file from JupyterLab's
# file browser. Right-click `artifacts_qwen25_7b.tar.gz` → Download.

# %%
import subprocess
from pathlib import Path

# Only include directories that actually exist (handles SFT-only / GRPO-only runs)
to_archive = [p for p in ["outputs/qwen25_7b_sft_final", "outputs/qwen25_7b_grpo_final", "eval/results"]
              if Path(p).exists()]

if not to_archive:
    print("⚠ Nothing to archive — no SFT, GRPO, or eval results found")
else:
    subprocess.check_call(["tar", "czf", "artifacts_qwen25_7b.tar.gz"] + to_archive)
    print(f"\n✓ Artifacts packaged: artifacts_qwen25_7b.tar.gz")
    print(f"\nContents:")
    for p in to_archive:
        print(f"  {p}")
    print(f"\nRight-click artifacts_qwen25_7b.tar.gz in JupyterLab's file panel → Download.")
    print("Then upload to your private HF repo.")
