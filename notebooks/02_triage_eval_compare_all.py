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
# # sre-gym Triage — compare every local artifact
#
# This notebook evaluates the Triage baselines plus every local adapter that is
# currently available in `outputs/`:
#
# - `qwen25-3b-sft`
# - `qwen25-3b-grpo`
# - `qwen25-7b-sft`
# - `qwen25-7b-grpo`
#
# Missing artifacts are skipped automatically, so you can re-run this notebook
# after each new training run and keep one rolling comparison table.

# %% [markdown]
# ## Cell 0 — Bootstrap
#
# Same repo bootstrap pattern as the training notebooks. Run this first after
# every kernel restart.

# %%
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

GITHUB_USER = "Madhav-GPT"
REPO_NAME = "sre-env"
BRANCH = "main"

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


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


REQUIRED = [
    "unsloth",
    "trl",
    "vllm",
    "datasets",
    "transformers",
    "matplotlib",
    "pandas",
    "httpx",
    "fastapi",
    "openenv",
    "uvicorn",
    "websockets",
    "yaml",
]

if all(_have(p) for p in REQUIRED) and _have("unified_incident_env"):
    print("✓ All deps already installed — skipping")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-qqq", "uv"])
    try:
        import numpy

        get_numpy = f"numpy=={numpy.__version__}"
    except ImportError:
        get_numpy = "numpy"
    subprocess.check_call(
        [
            "uv",
            "pip",
            "install",
            "-qqq",
            "--system",
            "torch>=2.8.0",
            "triton>=3.4.0",
            get_numpy,
            "torchvision",
            "bitsandbytes",
            "transformers==4.56.2",
            "trackio",
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
            "unsloth[base] @ git+https://github.com/unslothai/unsloth",
            "vllm",
            "datasets",
            "accelerate",
            "matplotlib",
            "pandas",
            "httpx",
            "fastapi",
            "pydantic>=2.0",
            "openenv-core>=0.2.1",
            "uvicorn[standard]>=0.30.0",
            "websockets>=12.0",
            "pyyaml>=6.0",
            "rich>=13.0.0",
        ]
    )
    subprocess.check_call(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--upgrade",
            "--no-deps",
            "transformers==4.56.2",
            "tokenizers",
            "trl==0.22.2",
            "unsloth",
            "unsloth_zoo",
        ]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."])
    print("✓ Installed notebook dependencies")

# %% [markdown]
# ## Cell 1 — Discover local adapters

# %%
import json
from pathlib import Path

MODEL_SOURCES = [
    {
        "label": "qwen25-3b-sft",
        "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "adapter_dir": Path("outputs/sft_final"),
        "max_seq_length": 4096,
    },
    {
        "label": "qwen25-3b-grpo",
        "base_model": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "adapter_dir": Path("outputs/grpo_final"),
        "max_seq_length": 4096,
    },
    {
        "label": "qwen25-7b-sft",
        "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "adapter_dir": Path("outputs/qwen25_7b_sft_final"),
        "max_seq_length": 2048,
    },
    {
        "label": "qwen25-7b-grpo",
        "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "adapter_dir": Path("outputs/qwen25_7b_grpo_final"),
        "max_seq_length": 2048,
    },
]

AVAILABLE_MODELS = [
    spec for spec in MODEL_SOURCES if (spec["adapter_dir"] / "adapter_model.safetensors").exists()
]

holdout = json.load(open("eval/holdout_basic.json"))
HOLDOUT_SCENARIOS = holdout["scenario_ids"]
SEEDS = [0, 1, 2]
RESULTS_CSV = Path("eval/results/triage_compare_all_raw.csv")

print(f"Holdout scenarios: {len(HOLDOUT_SCENARIOS)}")
print("Available adapters:")
for spec in AVAILABLE_MODELS:
    print(f"  - {spec['label']}: {spec['adapter_dir']}")
if not AVAILABLE_MODELS:
    print("  (none found — the notebook will still run baseline-only eval)")

# %% [markdown]
# ## Cell 2 — Run baselines and local models

# %%
import gc
import random

import pandas as pd
import torch
from peft import PeftModel
from unsloth import FastLanguageModel

from unified_incident_env.models import UnifiedIncidentAction
from unified_incident_env.server.challenge import list_baselines
from unified_incident_env.server.environment import UnifiedIncidentEnvironment

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


def load_local_adapter(spec: dict):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base_model"],
        max_seq_length=spec["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = PeftModel.from_pretrained(model, str(spec["adapter_dir"]), is_trainable=False)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def extract_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.startswith("```")).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def lm_action(prompt: str, lm, tok, max_new: int = 120):
    ids = tok.apply_chat_template(
        [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(lm.device)
    out = lm.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    return extract_json(text) or {"action_type": "escalate"}


def run_lm(scenario_id: str, seed: int, lm, tok, max_steps: int = 15):
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id, seed=seed)
    for _ in range(max_steps):
        data = lm_action(obs.prompt_text or "", lm, tok)
        try:
            action = UnifiedIncidentAction(**data)
        except Exception:
            action = UnifiedIncidentAction(action_type="escalate")
        obs = env.step(action)
        if obs.done:
            break
    return obs


def run_callable(scenario_id: str, seed: int, policy, max_steps: int = 15):
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id, seed=seed)
    for _ in range(max_steps):
        obs = env.step(policy(env, obs))
        if obs.done:
            break
    return obs


def random_policy(env, obs):
    rng = random.Random(env._episode["tick"] + hash(obs.prompt_text) % 1000)
    action_type = rng.choice(
        [
            "query_logs",
            "query_metrics",
            "query_dependencies",
            "query_deploys",
            "rollback_deploy",
            "restart_service",
            "isolate_service",
            "run_check",
            "declare_resolved",
            "escalate",
        ]
    )
    payload = {"action_type": action_type}
    if action_type in {
        "query_logs",
        "query_dependencies",
        "query_deploys",
        "rollback_deploy",
        "restart_service",
        "isolate_service",
    }:
        payload["service"] = rng.choice(["api-gateway", "cache", "database", "worker"])
    if action_type == "query_metrics":
        payload["service"] = rng.choice(["api-gateway", "cache", "database", "worker"])
        payload["metric"] = rng.choice(["cpu", "error_rate", "latency"])
    if action_type == "run_check":
        payload["check_name"] = rng.choice(["database_recovery", "end_to_end"])
    try:
        return UnifiedIncidentAction(**payload)
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


def scripted_for(scenario_id: str):
    actions = [row.action for row in list_baselines(scenario_id=scenario_id).baselines[0].actions]
    cursor = {"i": 0}

    def policy(env, obs):
        if cursor["i"] >= len(actions):
            return UnifiedIncidentAction(action_type="escalate")
        action = actions[cursor["i"]]
        cursor["i"] += 1
        return action

    return policy


def row(policy: str, scenario_id: str, seed: int, obs):
    return {
        "policy": policy,
        "scenario_id": scenario_id,
        "seed": seed,
        "final_score": obs.final_score,
        "incident_resolved": obs.incident_resolved,
        "steps": obs.tick_count,
    }


results = []

print("Running baselines ...")
for scenario_id in HOLDOUT_SCENARIOS:
    print(f"  baselines :: {scenario_id}")
    for seed in SEEDS:
        results.append(row("random", scenario_id, seed, run_callable(scenario_id, seed, random_policy)))
        results.append(row("heuristic", scenario_id, seed, run_callable(scenario_id, seed, heuristic_policy)))
        results.append(row("scripted_optimal", scenario_id, seed, run_callable(scenario_id, seed, scripted_for(scenario_id))))

for spec in AVAILABLE_MODELS:
    print(f"Running model eval :: {spec['label']}")
    model, tokenizer = load_local_adapter(spec)
    try:
        for scenario_id in HOLDOUT_SCENARIOS:
            print(f"  {spec['label']} :: {scenario_id}")
            for seed in SEEDS:
                results.append(row(spec["label"], scenario_id, seed, run_lm(scenario_id, seed, model, tokenizer)))
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"✓ Saved {len(results_df)} rows to {RESULTS_CSV}")

# %% [markdown]
# ## Cell 3 — Summary table + plots

# %%
import matplotlib.pyplot as plt
import pandas as pd

results_df = pd.read_csv("eval/results/triage_compare_all_raw.csv")

summary = results_df.groupby("policy").agg(
    mean=("final_score", "mean"),
    median=("final_score", "median"),
    p25=("final_score", lambda x: x.quantile(0.25)),
    p75=("final_score", lambda x: x.quantile(0.75)),
    resolved_rate=("incident_resolved", "mean"),
).round(3).sort_values("mean")

summary.to_csv("eval/results/triage_compare_all_summary.csv")
print(summary)

fig, ax = plt.subplots(figsize=(10, 5))
order = summary.index.tolist()
ax.bar(
    order,
    summary["mean"],
    yerr=[summary["mean"] - summary["p25"], summary["p75"] - summary["mean"]],
    capsize=5,
    color="#3a86ff",
)
ax.axhline(0.65, ls="--", color="gray", alpha=0.5, label="heuristic floor (0.65)")
ax.axhline(0.80, ls="--", color="gray", alpha=0.5, label="heuristic ceiling (0.80)")
ax.axhline(0.90, ls="--", color="green", alpha=0.5, label="scripted reference (0.90)")
ax.set_ylabel("Final score")
ax.set_xlabel("Policy")
ax.set_title("sre-gym Triage compare-all holdout eval")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", fontsize=8)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("eval/results/triage_compare_all_hero.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(13, 6))
template_ids = sorted({scenario_id.split("__")[0] for scenario_id in HOLDOUT_SCENARIOS})
positions = list(range(len(template_ids)))
policies_present = list(summary.index)
bar_width = 0.8 / max(len(policies_present), 1)
for offset, policy in enumerate(policies_present):
    sub = results_df[results_df.policy == policy]
    means = [sub[sub.scenario_id.str.startswith(template)]["final_score"].mean() for template in template_ids]
    ax.bar([p + offset * bar_width for p in positions], means, width=bar_width, label=policy)
ax.set_xticks([p + bar_width * len(policies_present) / 2 for p in positions])
ax.set_xticklabels(template_ids, rotation=30, ha="right")
ax.set_ylabel("Mean final_score")
ax.set_title("Per-template mean score by policy")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("eval/results/triage_compare_all_per_template.png", dpi=150)
plt.show()
