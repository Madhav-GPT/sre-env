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
# # sre-gym Triage — Qwen2.5-3B SFT → GRPO training notebook
#
# **Target:** A100 80GB · ~2.5h wall-clock · ~$6 in HF compute credits.
#
# ## How to run this notebook
#
# 1. **Run Cell 0 first** (idempotent — safe to re-run after any kernel restart, takes ~5s if already set up).
# 2. Then run cells 1–12 top to bottom.
# 3. **After any "Kernel → Restart"**, you only need to re-run Cell 0 + the cell you want to continue from. Every cell imports what it needs.
# 4. If a cell fails, read the printed **FIX** message — it tells you exactly what to do.
#
# ## Resume points
# - If `outputs/sft_final/` exists → SFT cells skip and load from disk.
# - If `outputs/grpo_final/` exists → GRPO cell skips and loads from disk.
# - To force re-train, delete the corresponding output dir.

# %% [markdown]
# ## Cell 0 — Bootstrap (RUN ME FIRST after every kernel restart)
#
# Idempotent. Detects state and only does what's needed:
# - First run: clone repo, install deps (~3 min)
# - After kernel restart: re-import, re-chdir (~5 sec)
# - Re-run within same session: no-op (<1 sec)

# %%
import os
import subprocess
import sys
from pathlib import Path

GITHUB_USER = "Madhav-GPT"
REPO_NAME   = "sre-env"
BRANCH      = "main"

# ---- Step 1: ensure cwd is the repo root ----
if Path("sre_gym").exists() and Path("notebooks").exists():
    print(f"Already in repo root: {Path('.').resolve()}")
elif Path(REPO_NAME).exists():
    os.chdir(REPO_NAME)
    print(f"Changed to repo root: {Path('.').resolve()}")
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
    print(f"Cloned to: {Path('.').resolve()}")

REPO_ROOT = Path(".").resolve()
assert (REPO_ROOT / "sre_gym").exists(), "Wrong cwd — sre_gym/ not found"

# Make repo importable from anywhere
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Step 2: install deps (idempotent — checks before installing) ----
def _all_installed(pkgs):
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            return False
    return True

REQUIRED = ["unsloth", "trl", "vllm", "datasets", "transformers", "matplotlib", "pandas"]
if _all_installed(REQUIRED):
    print("All deps already installed — skipping pip install")
else:
    # Fallback path for users who haven't pre-installed via Unsloth's pattern.
    # If you're using Unsloth's official install in a separate cell, this branch
    # never runs.
    print("Installing deps — this is 8-15 min on first run.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                            "unsloth", "trl", "vllm",
                            "datasets", "accelerate", "matplotlib", "pandas",
                            "httpx", "fastapi", "pydantic>=2.0"])
    print("\n✓ Deps installed")

# ---- Step 3: GPU sanity check via nvidia-smi (before importing torch) ----
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
    capture_output=True, text=True,
)
if gpu.returncode != 0:
    raise RuntimeError(
        "\n\n  nvidia-smi failed — this Space has NO GPU.\n"
        "  FIX: Space → Settings → 'Space hardware' → 'Nvidia A100 Large 80GB' → Save\n"
        "  Then wait ~2 min for the Space to restart and re-run this cell.\n"
    )
print(f"\nGPU: {gpu.stdout.strip()}")
print(f"Repo root: {REPO_ROOT}")
print("✓ Cell 0 complete — proceed to Cell 1")

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
        f"  FIX: CUDA version mismatch. In a NEW cell run:\n"
        f"    !pip uninstall -y torch torchvision torchaudio\n"
        f"    !pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121\n"
        f"  Then: Kernel → Restart Kernel → re-run Cell 0 → re-run Cell 1.\n"
    )

device_name = torch.cuda.get_device_name(0)
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f"GPU:        {device_name}")
print(f"VRAM:       {vram_gb} GB")
print(f"PyTorch:    {torch.__version__} (CUDA {torch.version.cuda})")

if vram_gb < 70:
    raise RuntimeError(
        f"\n  Need ≥70 GB VRAM for Qwen2.5-3B + K=4 GRPO; got {vram_gb} GB.\n"
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
# Drops steps with empty/short prompts, non-JSON responses, or >4096 tokens.

# %%
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_SEQ_LEN = 4096

tokenizer_for_chat = AutoTokenizer.from_pretrained(MODEL_NAME)

SFT_SYSTEM_PROMPT = """You are a senior SRE on-call agent inside the sre-gym Triage environment.

Output EXACTLY one JSON object per turn — no prose, no markdown, no fences.
The 11 actions are:
  query_logs(service)            query_metrics(service, metric)
  query_dependencies(service)    query_deploys(service)
  rollback_deploy(service)       restart_service(service)
  isolate_service(service)       run_check(check_name)
  submit_hypothesis(hypothesis)  escalate
  declare_resolved

Services: api-gateway / cache / database / worker.
metric in {cpu, error_rate, latency}; check_name in {database_recovery, end_to_end}.

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
# ## Cell 5 — Load Qwen2.5-3B (4-bit + LoRA r=64)
#
# Skips download if already cached. ~3 min cold, <30 sec warm.

# %%
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_SEQ_LEN = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)
print("Trainable params:")
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 6 — SFT cold-start (150 steps) + perplexity gate
#
# **Resume:** if `outputs/sft_final/` exists, skips training and loads from disk.
# To force re-train, delete the directory.

# %%
import math
from pathlib import Path

from trl import SFTTrainer, SFTConfig

SFT_OUT = Path("outputs/sft_final")

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
        output_dir="outputs/sft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=150,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
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
    print(f"\nFinal eval perplexity: {final_perplexity:.3f}")

    if final_perplexity < 1.2:
        raise RuntimeError(
            f"\n  Perplexity {final_perplexity:.3f} < 1.2 — policy collapsed to determinism.\n"
            f"  FIX: Skip GRPO. Ship outputs/sft_final/ as-is, OR add data diversity & retrain."
        )
    if final_perplexity > 4.0:
        raise RuntimeError(
            f"\n  Perplexity {final_perplexity:.3f} > 4.0 — SFT undercooked.\n"
            f"  FIX: Delete outputs/sft_final/, bump max_steps to 250, re-run Cell 6."
        )
    if 1.5 <= final_perplexity <= 2.5:
        print("✓ Perplexity in healthy band [1.5, 2.5] — proceed to GRPO")
    else:
        print(f"⚠ Perplexity {final_perplexity:.3f} outside ideal [1.5, 2.5] but acceptable [1.2, 4.0]")

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


def build_grpo_prompts(num_prompts=120, seed=0):
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
        out.append({"prompt": obs.prompt_text, "scenario_id": scenario})
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


def reward_fn(completions, prompts=None, **kwargs):
    """Per-turn proxy reward. Returns one float per completion."""
    scenario_ids = kwargs.get("scenario_id") or [None] * len(completions)
    rewards = []
    for completion, scenario_id in zip(completions, scenario_ids):
        if scenario_id is None:
            rewards.append(0.0)
            continue
        action_dict = _extract_action_json(completion)
        if action_dict is None:
            rewards.append(-0.5)
            continue
        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception:
            rewards.append(-0.3)
            continue
        env = UnifiedIncidentEnvironment()
        try:
            env.reset(scenario_id=scenario_id)
            obs = env.step(action)
        except Exception:
            rewards.append(-0.2)
            continue
        r = float(obs.reward)
        if obs.failure_type:
            r -= 0.2
        if obs.incident_resolved:
            r += 0.5
        rewards.append(r)
    return rewards


# Smoke test
_test = reward_fn(['{"action_type":"query_deploys","service":"worker"}'],
                  scenario_id=["worker_deploy_cascade"])
print(f"Smoke test reward: {_test[0]:+.3f}")
assert _test[0] > -0.5, "reward_fn smoke test failed"
print("✓ reward_fn validated")

# %% [markdown]
# ## Cell 9 — GRPO online training (100 steps × K=4)
#
# **Resume:** if `outputs/grpo_final/` exists, skips training.
# **Fallback:** if vLLM crashes (common on some Spaces), retries with `use_vllm=False` (~3× slower but works).

# %%
from pathlib import Path

from trl import GRPOTrainer, GRPOConfig

GRPO_OUT = Path("outputs/grpo_final")


def _build_grpo_args(use_vllm: bool):
    # TRL 0.22.2 (Unsloth's recommended): max_prompt_length exists, vllm_max_model_length doesn't.
    # TRL 1.x: vllm_max_model_length exists, max_prompt_length removed. We target 0.22.2.
    return GRPOConfig(
        output_dir="outputs/grpo",
        num_generations=4,
        max_steps=100,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_prompt_length=2048,
        max_completion_length=256,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=0.5 if use_vllm else 0.0,
        beta=0.04,
        temperature=0.7,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        report_to="none",
    )


if GRPO_OUT.exists() and (GRPO_OUT / "adapter_model.safetensors").exists():
    print(f"✓ GRPO already trained at {GRPO_OUT} — loading adapter weights")
    from peft import PeftModel
    # If model already has SFT LoRA loaded, swap to GRPO
    if hasattr(model, "load_adapter"):
        model.load_adapter(str(GRPO_OUT), adapter_name="default")
    print("Loaded GRPO adapter — skipping GRPO training")
else:
    print("Starting GRPO online training ...")
    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            args=_build_grpo_args(use_vllm=True),
            reward_funcs=[reward_fn],
            train_dataset=grpo_prompts_ds,
            processing_class=tokenizer,
        )
        grpo_trainer.train()
    except Exception as exc:
        print(f"\n⚠ vLLM path failed: {exc}")
        print("Retrying without vLLM (slower but more compatible) ...")
        grpo_trainer = GRPOTrainer(
            model=model,
            args=_build_grpo_args(use_vllm=False),
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
# 5 policies × 12 holdout × 3 seeds = 180 episodes. Saves to
# `eval/results/comparison_raw.csv`. Skips if results already exist.

# %%
import json
import random
import sys
from pathlib import Path

import pandas as pd
from unsloth import FastLanguageModel

if str(Path(".").resolve()) not in sys.path:
    sys.path.insert(0, str(Path(".").resolve()))

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.environment import UnifiedIncidentEnvironment
from unified_incident_env.server.challenge import list_baselines

EVAL_CSV = Path("eval/results/comparison_raw.csv")
MAX_SEQ_LEN = 4096

if EVAL_CSV.exists():
    results_df = pd.read_csv(EVAL_CSV)
    print(f"✓ Eval already done — loaded {len(results_df)} rows from {EVAL_CSV}")
else:
    holdout = json.load(open("eval/holdout_basic.json"))
    HOLDOUT_SCENARIOS = holdout["scenario_ids"]
    print(f"Holdout: {len(HOLDOUT_SCENARIOS)} scenarios")

    # Switch GRPO model to inference mode
    FastLanguageModel.for_inference(model)

    # Load SFT-only adapter into a separate model
    sft_only_model, _ = FastLanguageModel.from_pretrained(
        model_name="outputs/sft_final",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(sft_only_model)

    def _extract_json(text):
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
        s = text.find("{"); e = text.rfind("}")
        if s < 0 or e <= s: return None
        try: return json.loads(text[s:e+1])
        except: return None

    def lm_action(prompt, lm, max_new=120):
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": SFT_SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(lm.device)
        out = lm.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return _extract_json(text) or {"action_type": "escalate"}

    def run_lm(scenario_id, seed, lm, max_steps=15):
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=scenario_id)
        for _ in range(max_steps):
            d = lm_action(obs.prompt_text or "", lm)
            try: action = UnifiedIncidentAction(**d)
            except: action = UnifiedIncidentAction(action_type="escalate")
            obs = env.step(action)
            if obs.done: break
        return obs

    def run_callable(scenario_id, seed, policy, max_steps=15):
        env = UnifiedIncidentEnvironment()
        obs = env.reset(scenario_id=scenario_id)
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

    results = []
    for sid in HOLDOUT_SCENARIOS:
        print(f"  {sid} ...")
        for seed in range(3):
            obs = run_callable(sid, seed, random_policy)
            results.append({"policy":"random","scenario_id":sid,"seed":seed,
                            "final_score":obs.final_score,"incident_resolved":obs.incident_resolved,"steps":obs.tick_count})
            obs = run_callable(sid, seed, heuristic_policy)
            results.append({"policy":"heuristic","scenario_id":sid,"seed":seed,
                            "final_score":obs.final_score,"incident_resolved":obs.incident_resolved,"steps":obs.tick_count})
            obs = run_callable(sid, seed, scripted_for(sid))
            results.append({"policy":"scripted_optimal","scenario_id":sid,"seed":seed,
                            "final_score":obs.final_score,"incident_resolved":obs.incident_resolved,"steps":obs.tick_count})
            obs = run_lm(sid, seed, sft_only_model)
            results.append({"policy":"qwen25-3b-sft-only","scenario_id":sid,"seed":seed,
                            "final_score":obs.final_score,"incident_resolved":obs.incident_resolved,"steps":obs.tick_count})
            obs = run_lm(sid, seed, model)
            results.append({"policy":"qwen25-3b-grpo","scenario_id":sid,"seed":seed,
                            "final_score":obs.final_score,"incident_resolved":obs.incident_resolved,"steps":obs.tick_count})

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

results_df = pd.read_csv("eval/results/comparison_raw.csv")
holdout = json.load(open("eval/holdout_basic.json"))
HOLDOUT_SCENARIOS = holdout["scenario_ids"]

summary = results_df.groupby("policy").agg(
    mean=("final_score", "mean"),
    median=("final_score", "median"),
    p25=("final_score", lambda x: x.quantile(0.25)),
    p75=("final_score", lambda x: x.quantile(0.75)),
    resolved_rate=("incident_resolved", "mean"),
).round(3).sort_values("mean")
summary.to_csv("eval/results/comparison_summary.csv")
print(summary)

fig, ax = plt.subplots(figsize=(9, 5))
order = summary.index.tolist()
ax.bar(order, summary["mean"],
       yerr=[summary["mean"] - summary["p25"], summary["p75"] - summary["mean"]],
       capsize=5, color="#3a86ff")
ax.axhline(0.65, ls="--", color="gray", alpha=0.5, label="heuristic floor (0.65)")
ax.axhline(0.80, ls="--", color="gray", alpha=0.5, label="heuristic ceiling (0.80)")
ax.axhline(0.90, ls="--", color="green", alpha=0.5, label="scripted reference (0.90)")
ax.set_ylabel("Final score (5-component composite)")
ax.set_xlabel("Policy")
ax.set_title("sre-gym Triage holdout eval (12 scenarios × 3 seeds)")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", fontsize=8)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("eval/results/comparison_hero.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(13, 6))
template_ids = sorted({s.split("__")[0] for s in HOLDOUT_SCENARIOS})
positions = list(range(len(template_ids)))
for offset, policy in enumerate(["random","heuristic","scripted_optimal","qwen25-3b-sft-only","qwen25-3b-grpo"]):
    sub = results_df[results_df.policy == policy]
    means = [sub[sub.scenario_id.str.startswith(t)]["final_score"].mean() for t in template_ids]
    ax.bar([p + offset * 0.15 for p in positions], means, width=0.15, label=policy)
ax.set_xticks([p + 0.30 for p in positions])
ax.set_xticklabels(template_ids, rotation=30, ha="right")
ax.set_ylabel("Mean final_score")
ax.legend(fontsize=8)
ax.set_title("Per-template mean score by policy")
plt.tight_layout()
plt.savefig("eval/results/comparison_per_template.png", dpi=150)
plt.show()

# %% [markdown]
# ## Cell 12 — Package artifacts for download
#
# Tars the final artifacts so you can download a single file from JupyterLab's
# file browser. Right-click `artifacts.tar.gz` → Download.

# %%
import subprocess

subprocess.check_call([
    "tar", "czf", "artifacts.tar.gz",
    "outputs/sft_final",
    "outputs/grpo_final",
    "eval/results",
])
print("\n✓ Artifacts packaged: artifacts.tar.gz")
print()
print("Contents:")
print("  outputs/sft_final/    — SFT-only LoRA adapter")
print("  outputs/grpo_final/   — GRPO-trained LoRA adapter (the headline result)")
print("  eval/results/         — comparison_raw.csv, summary.csv, hero.png, per_template.png")
print()
print("Right-click artifacts.tar.gz in JupyterLab's file panel → Download.")
print("Then upload to your private HF repo.")
