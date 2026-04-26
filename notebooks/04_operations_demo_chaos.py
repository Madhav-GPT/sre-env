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
# # sre-gym Operations tier — chaos and family walkthrough
#
# `Operations` is the canonical name for the old `Max` tier. This notebook
# inspects the shipped family spec and chaos library so you can see what the
# realism-bounded tier looks like without provisioning the full stack.

# %% [markdown]
# ## Cell 0 — Bootstrap

# %%
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

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
    else:
        url = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"
    subprocess.check_call(["git", "clone", "--depth=1", "--branch", BRANCH, url, REPO_NAME])
    os.chdir(REPO_NAME)

REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


REQUIRED = ["yaml", "pandas", "fastapi", "pydantic", "openenv"]
if not all(_have(pkg) for pkg in REQUIRED):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pyyaml>=6.0",
            "pandas",
            "openenv-core>=0.2.1",
            "fastapi>=0.115",
            "pydantic>=2.8",
            "matplotlib",
        ]
    )

print(f"Repo root: {REPO_ROOT}")

# %% [markdown]
# ## Cell 1 — Tier metadata

# %%
from sre_gym import SREGym
from sre_gym.tier import TIER_CONFIGS, Tier

env = SREGym(tier=Tier.OPERATIONS)
cfg = TIER_CONFIGS[Tier.OPERATIONS]

print(env.describe())
print("\nCanonical tier config:")
for key, value in cfg.__dict__.items():
    print(f"  {key:<28} {value}")

# %% [markdown]
# ## Cell 2 — Family spec overview

# %%
import pandas as pd
import yaml
from IPython.display import display

FAMILY_PATH = REPO_ROOT / "sre_gym" / "operations" / "families" / "ecommerce_vibecoded_saas.yaml"
CHAOS_PATH = REPO_ROOT / "sre_gym" / "operations" / "chaos" / "ecommerce_chaos_library.yaml"


def load_yaml(path: Path):
    with path.open() as handle:
        return yaml.safe_load(handle)


family = load_yaml(FAMILY_PATH)
topology_df = pd.DataFrame(family["topology"]["services"])
composition_df = pd.DataFrame(family["scenario_population"]["composition_rules"])
rubric_df = pd.DataFrame(
    [{"dimension": key, **value} for key, value in family["reward_model"]["rubric_dimensions"].items()]
)

print(f"Family: {family['id']}")
print(f"Tier  : {family['tier']}")
print(f"\nDescription:\n{family['description']}")

print("\nTopology counts by kind:")
display(topology_df.groupby("kind").size().rename("count").reset_index())

print("\nScenario population:")
display(composition_df)

print("\nReward rubric dimensions:")
display(rubric_df.sort_values("dimension").reset_index(drop=True))

# %% [markdown]
# ## Cell 3 — Reference instance and workload

# %%
workload_df = pd.DataFrame(
    [
        {"endpoint": endpoint, **settings}
        for endpoint, settings in family["workload_generator"]["rates"].items()
    ]
)

reference = family["reference_instance"]

print(f"Reference instance: {reference['id']}")
print(f"\nDescription:\n{reference['description']}")
print("\nChaos patterns applied:")
for pattern in reference["chaos_patterns_applied"]:
    print(f"  - {pattern}")

print("\nWorkload generator:")
display(workload_df)

print("\nOperator notes:")
for key, value in family["operator_notes"].items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Cell 4 — Chaos library

# %%
chaos = load_yaml(CHAOS_PATH)
pattern_rows = []
for pattern_name, pattern_spec in chaos["patterns"].items():
    pattern_rows.append(
        {
            "pattern": pattern_name,
            "targets": ", ".join(pattern_spec.get("targets", [])),
            "inject_type": pattern_spec.get("inject", {}).get("type"),
            "grader_focus": pattern_spec.get("grader_focus"),
            "classification": pattern_spec.get("classification", "reliability"),
        }
    )

patterns_df = pd.DataFrame(pattern_rows).sort_values("pattern")
display(patterns_df)

print("\nComposition safety:")
for key, value in chaos["composition_safety"].items():
    print(f"  {key}: {value}")

print(
    "\nInterpretation: Operations keeps the same high-level incident goal as Triage, "
    "but the environment shifts to live-ish tool use, multi-fault composition, and "
    "outcome-based scoring over a much richer topology."
)
