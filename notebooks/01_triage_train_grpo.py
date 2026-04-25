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
# Pipeline:
# 1. **Bootstrap** — clone repo, install deps. (Cell 0)
# 2. **Build SFT corpus** — 60% expert / 20% mediocre / 20% failure (120 episodes).
# 3. **SFT cold-start** — 150 steps, LoRA r=64. Target eval-perplexity 1.5–2.5.
# 4. **GRPO online** — 100 steps, K=4 rollouts/prompt, per-turn proxy reward.
# 5. **Eval sweep** — 5 policies × 12 holdout × 3 seeds = 180 runs.
# 6. **Plots + summary** → `eval/results/`.
#
# **Stop gate at Cell 6:** if SFT eval perplexity falls outside [1.2, 4.0], do not
# proceed to GRPO — fix the data first.

# %% [markdown]
# ## Cell 0 — Bootstrap
#
# Clones the repo, installs deps. Skips clone if the repo is already present
# (e.g. you've re-run the notebook). Adjust `REPO_URL` / `BRANCH` if you forked.

# %%
# !pip install -q --upgrade pip
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/dakshdoesdev/sre-enginnerllm.git"
REPO_DIR = "sre-env"
BRANCH = "main"

if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", "--depth=1", "--branch", BRANCH, REPO_URL, REPO_DIR])
else:
    print(f"{REPO_DIR}/ already exists — skipping clone")

os.chdir(REPO_DIR)
REPO_ROOT = Path(".").resolve()
print("Repo root:", REPO_ROOT)
assert (REPO_ROOT / "sre_gym").exists(), "Wrong directory — sre_gym/ not found"

# Install repo + deps. Unsloth/TRL/vLLM are pulled fresh because the wheels are
# CUDA-version-pinned and the Space's torch may differ from the lockfile's.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,train]"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "unsloth", "trl>=0.11", "vllm", "datasets", "accelerate",
                        "matplotlib", "pandas"])

# Make the repo importable for in-process env stepping later.
sys.path.insert(0, str(REPO_ROOT))

# %% [markdown]
# ## Cell 1 — GPU sanity check

# %%
import torch

assert torch.cuda.is_available(), "No GPU detected — switch the Space hardware to A100 large"
print("GPU:", torch.cuda.get_device_name(0))
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print("VRAM:", vram_gb, "GB")
assert vram_gb > 70, "Need an 80GB A100 for the 3B + K=4 path (this looks like a smaller card)"

# %% [markdown]
# ## Cell 2 — Build the SFT corpus
#
# Calls `train/build_corpus.py`. With `train/data/sonnet_missing6.jsonl` already
# in the repo (committed from your laptop), the corpus reaches the full
# 120-episode target. Without it, you'll get ~115 — still trainable, slightly
# weaker on the 6 missing templates.

# %%
import json

result = subprocess.run(
    [sys.executable, "train/build_corpus.py", "--output", "train/data/seed_v2_120.jsonl"],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    raise RuntimeError("build_corpus.py failed — see stderr above")

# %% [markdown]
# ## Cell 3 — Sanity-check the corpus
#
# Verify the 60/20/20 quality split, template-uniform coverage, score
# distribution. If any assertion fails, fix the corpus before training.

# %%
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

# %% [markdown]
# ## Cell 4 — Convert each step into a (prompt, completion) ChatML pair
#
# Each trajectory step becomes one SFT example. Drop steps with empty prompts
# or non-JSON responses, and steps whose ChatML exceeds 4096 tokens.

# %%
from transformers import AutoTokenizer

MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
MAX_SEQ_LEN = 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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


def step_to_chatml(prompt: str, response: str) -> str:
    return tokenizer.apply_chat_template(
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
            tokens = tokenizer(text, truncation=False, return_length=True)["length"][0]
            if tokens > MAX_SEQ_LEN:
                continue
            pairs.append({"text": text, "tokens": tokens, "tier": ep["quality_tier"]})

print(f"SFT step-pairs: {len(pairs)}")
print("Token length distribution:")
print(pd.Series([p['tokens'] for p in pairs]).describe().round(0))

from datasets import Dataset
sft_dataset = Dataset.from_list([{"text": p["text"]} for p in pairs])
sft_dataset = sft_dataset.shuffle(seed=42)
print("\nDataset:", sft_dataset)

# %% [markdown]
# ## Cell 5 — Load Qwen2.5-3B with Unsloth (4-bit + LoRA r=64)

# %%
from unsloth import FastLanguageModel

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
# Watch `eval_loss`. The handover criterion is validation perplexity in
# [1.5, 2.5]. If perplexity drops below 1.2 the policy has collapsed to
# determinism — **do not proceed to GRPO**, save SFT-only and ship that.
# Above 4.0 means SFT undercooked — bump `max_steps` and rerun.

# %%
from trl import SFTTrainer, SFTConfig

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
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataset_text_field="text",
)
sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=sft_args,
)
print("\nStarting SFT cold-start ...")
sft_trainer.train()
sft_trainer.save_model("outputs/sft_final")
print("Saved to outputs/sft_final")

# %%
import math

eval_metrics = sft_trainer.evaluate()
print("Final eval metrics:", eval_metrics)
final_perplexity = math.exp(eval_metrics["eval_loss"])
print(f"\nFinal eval perplexity: {final_perplexity:.3f}")

if final_perplexity < 1.2:
    raise RuntimeError(
        f"Perplexity {final_perplexity:.3f} < 1.2 — policy collapsed to determinism. "
        f"Skip GRPO. Save outputs/sft_final and either (a) ship SFT-only, "
        f"or (b) re-run with more diverse trajectories."
    )
elif final_perplexity > 4.0:
    raise RuntimeError(
        f"Perplexity {final_perplexity:.3f} > 4.0 — SFT undercooked. "
        f"Bump max_steps to 250 and rerun Cell 6."
    )
elif 1.5 <= final_perplexity <= 2.5:
    print("✓ Perplexity in healthy band [1.5, 2.5] — proceed to GRPO")
else:
    print(f"⚠ Perplexity {final_perplexity:.3f} outside ideal [1.5, 2.5] but within [1.2, 4.0] — proceed with caution")

# %% [markdown]
# ## Cell 7 — Build the GRPO prompt dataset
#
# Sample observations from the env after a varying number of warmup steps so
# the model sees prompts at different workflow stages (triage / mitigation).
# Each prompt carries `scenario_id` so the reward function can replay the env
# to the same state. **All env work is in-process Python — no HTTP** (the
# openenv HTTP server creates a fresh env per request, which would silently
# score every action against the default scenario).

# %%
import random

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
        if i >= len(templates):
            scenario = f"{base}__p0{1 + (i // len(templates)) % 4}"
        else:
            scenario = base
        env = UnifiedIncidentEnvironment()
        try:
            obs = env.reset(scenario_id=scenario)
        except Exception:
            scenario = base
            obs = env.reset(scenario_id=scenario)
        # Optional warmup so prompts span workflow stages
        if rng.random() > 0.5:
            obs = env.step(UnifiedIncidentAction(action_type="query_logs", service="worker"))
        out.append({
            "prompt": obs.prompt_text,
            "scenario_id": scenario,
        })
    return out


grpo_prompts_list = build_grpo_prompts(num_prompts=120, seed=42)
grpo_prompts_ds = Dataset.from_list(grpo_prompts_list)
print(f"GRPO prompts: {len(grpo_prompts_ds)}")
print("Sample scenario:", grpo_prompts_ds[0]["scenario_id"])
print("Sample prompt (truncated):", grpo_prompts_ds[0]["prompt"][:200], "...")

# %% [markdown]
# ## Cell 8 — Per-turn proxy reward function
#
# For each generated completion (one JSON action):
# 1. Parse the JSON (format penalty if invalid)
# 2. Reset a fresh env to the prompt's scenario
# 3. Step once with the parsed action
# 4. Return the env's per-tick shaped reward + bonuses

# %%
def _extract_action_json(text: str):
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
            rewards.append(-0.5)  # format penalty
            continue
        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception:
            rewards.append(-0.3)  # validation penalty
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


# Smoke test the reward function on a known-good action
_test_completion = '{"action_type":"query_deploys","service":"worker"}'
_test_kw = {"scenario_id": ["worker_deploy_cascade"]}
_test_r = reward_fn([_test_completion], **_test_kw)
print(f"Smoke test (query_deploys on worker_deploy_cascade): reward={_test_r[0]:.3f}")
assert _test_r[0] > -0.5, "reward_fn smoke test failed — investigate before running GRPO"

# %% [markdown]
# ## Cell 9 — GRPO online training (100 steps, K=4)
#
# Wall-clock budget: ~85 minutes on A100 with vLLM-backed sampling and
# `gpu_memory_utilization=0.5`. If the reward curve flatlines before step 100,
# stop early — most of the signal has been extracted.

# %%
from trl import GRPOTrainer, GRPOConfig

grpo_args = GRPOConfig(
    output_dir="outputs/grpo",
    num_generations=4,
    max_steps=100,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_prompt_length=2048,
    max_completion_length=256,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,
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
grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    reward_funcs=[reward_fn],
    train_dataset=grpo_prompts_ds,
    tokenizer=tokenizer,
)
print("\nStarting GRPO online training ...")
grpo_trainer.train()
grpo_trainer.save_model("outputs/grpo_final")
print("Saved to outputs/grpo_final")

# %% [markdown]
# ## Cell 10 — Eval comparison sweep
#
# 5 policies × 12 holdout scenarios × 3 seeds = 180 episodes. Each policy
# runs against the env in-process (no HTTP, no vLLM — uses the trained model
# via Unsloth's `FastLanguageModel.for_inference`).

# %%
holdout = json.load(open("eval/holdout_basic.json"))
HOLDOUT_SCENARIOS = holdout["scenario_ids"]
print(f"Holdout: {len(HOLDOUT_SCENARIOS)} scenarios")

FastLanguageModel.for_inference(model)


def lm_action(observation_prompt: str, model_to_use, max_new_tokens=120) -> dict:
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": observation_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model_to_use.device)
    out = model_to_use.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return _extract_action_json(text) or {"action_type": "escalate"}


def run_episode_with_lm(scenario_id, seed, lm_model, max_steps=15):
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    for _ in range(max_steps):
        action_dict = lm_action(obs.prompt_text or "", lm_model)
        try:
            action = UnifiedIncidentAction(**action_dict)
        except Exception:
            action = UnifiedIncidentAction(action_type="escalate")
        obs = env.step(action)
        if obs.done:
            break
    return obs


def run_episode_with_callable(scenario_id, seed, policy_callable, max_steps=15):
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    for _ in range(max_steps):
        action = policy_callable(env, obs)
        obs = env.step(action)
        if obs.done:
            break
    return obs


def random_policy(env, obs):
    rng = random.Random(env._episode["tick"] + hash(obs.prompt_text) % 1000)
    atype = rng.choice([
        "query_logs", "query_metrics", "query_dependencies", "query_deploys",
        "rollback_deploy", "restart_service", "isolate_service", "run_check",
        "declare_resolved", "escalate",
    ])
    kw = {"action_type": atype}
    if atype in {"query_logs", "query_dependencies", "query_deploys",
                 "rollback_deploy", "restart_service", "isolate_service"}:
        kw["service"] = rng.choice(["api-gateway", "cache", "database", "worker"])
    if atype == "query_metrics":
        kw["service"] = rng.choice(["api-gateway", "cache", "database", "worker"])
        kw["metric"] = rng.choice(["cpu", "error_rate", "latency"])
    if atype == "run_check":
        kw["check_name"] = rng.choice(["database_recovery", "end_to_end"])
    try:
        return UnifiedIncidentAction(**kw)
    except Exception:
        return UnifiedIncidentAction(action_type="escalate")


def heuristic_policy(env, obs):
    truth = env._episode["scenario"]["truth"]
    tick = env._episode["tick"]
    if tick == 0:
        return UnifiedIncidentAction(action_type="query_logs", service="worker")
    if tick == 1:
        return UnifiedIncidentAction(action_type="query_deploys", service="worker")
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


from unified_incident_env.server.challenge import list_baselines


def scripted_policy_for(scenario_id):
    actions = [step.action for step in list_baselines(scenario_id=scenario_id).baselines[0].actions]
    cursor = {"i": 0}
    def policy(env, obs):
        if cursor["i"] >= len(actions):
            return UnifiedIncidentAction(action_type="escalate")
        a = actions[cursor["i"]]
        cursor["i"] += 1
        return a
    return policy


# Load SFT-only model for comparison
sft_only_model, _ = FastLanguageModel.from_pretrained(
    model_name="outputs/sft_final",
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(sft_only_model)


results = []
for scenario_id in HOLDOUT_SCENARIOS:
    print(f"Evaluating {scenario_id} ...")
    for seed in range(3):
        # random
        obs = run_episode_with_callable(scenario_id, seed, random_policy)
        results.append({"policy": "random", "scenario_id": scenario_id, "seed": seed,
                        "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                        "steps": obs.tick_count})
        # heuristic
        obs = run_episode_with_callable(scenario_id, seed, heuristic_policy)
        results.append({"policy": "heuristic", "scenario_id": scenario_id, "seed": seed,
                        "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                        "steps": obs.tick_count})
        # scripted-optimal
        obs = run_episode_with_callable(scenario_id, seed, scripted_policy_for(scenario_id))
        results.append({"policy": "scripted_optimal", "scenario_id": scenario_id, "seed": seed,
                        "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                        "steps": obs.tick_count})
        # qwen2.5-3b-sft-only
        obs = run_episode_with_lm(scenario_id, seed, sft_only_model)
        results.append({"policy": "qwen25-3b-sft-only", "scenario_id": scenario_id, "seed": seed,
                        "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                        "steps": obs.tick_count})
        # qwen2.5-3b-grpo
        obs = run_episode_with_lm(scenario_id, seed, model)
        results.append({"policy": "qwen25-3b-grpo", "scenario_id": scenario_id, "seed": seed,
                        "final_score": obs.final_score, "incident_resolved": obs.incident_resolved,
                        "steps": obs.tick_count})

results_df = pd.DataFrame(results)
Path("eval/results").mkdir(parents=True, exist_ok=True)
results_df.to_csv("eval/results/comparison_raw.csv", index=False)
print(f"\nSaved {len(results_df)} eval rows to eval/results/comparison_raw.csv")

# %% [markdown]
# ## Cell 11 — Summary table + hero plot

# %%
import matplotlib.pyplot as plt

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
ax.bar(order, summary["mean"], yerr=[summary["mean"] - summary["p25"], summary["p75"] - summary["mean"]],
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
for offset, policy in enumerate(["random", "heuristic", "scripted_optimal", "qwen25-3b-sft-only", "qwen25-3b-grpo"]):
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
# file browser. Right-click `artifacts.tar.gz` in the file panel → Download.

# %%
subprocess.check_call([
    "tar", "czf", "artifacts.tar.gz",
    "outputs/sft_final",
    "outputs/grpo_final",
    "eval/results",
])
print("\nArtifacts packaged: artifacts.tar.gz")
print()
print("Contents:")
print("  outputs/sft_final/    — SFT-only LoRA adapter")
print("  outputs/grpo_final/   — GRPO-trained LoRA adapter (the headline result)")
print("  eval/results/         — comparison_raw.csv, summary.csv, hero.png, per_template.png")
print()
print("Right-click artifacts.tar.gz in JupyterLab's file panel → Download.")
print("Then upload to your private HF repo (Madhav189/model-train-env or wherever).")
