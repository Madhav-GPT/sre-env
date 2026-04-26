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
# # sre-gym Strategy tier — blueprint walkthrough
#
# `Strategy` is the repo's canonical name for the old `Advanced` tier. The
# code still accepts `Tier.ADVANCED` as an alias, but the tier itself is now
# documented as:
#
# - `Triage` — bounded by compute
# - `Strategy` — bounded by horizon
# - `Operations` — bounded by realism
#
# This notebook inspects the shipped Strategy YAML specs and shows what each
# scenario is teaching before you spend time building long-horizon training.

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
        ]
    )

print(f"Repo root: {REPO_ROOT}")

# %% [markdown]
# ## Cell 1 — Tier metadata

# %%
from sre_gym import SREGym
from sre_gym.tier import TIER_CONFIGS, Tier

env = SREGym(tier=Tier.STRATEGY)
cfg = TIER_CONFIGS[Tier.STRATEGY]

print(env.describe())
print("\nCanonical tier config:")
for key, value in cfg.__dict__.items():
    print(f"  {key:<28} {value}")

# %% [markdown]
# ## Cell 2 — Scenario catalog

# %%
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

SCENARIO_DIR = REPO_ROOT / "sre_gym" / "strategy" / "scenarios"


def load_yaml(path: Path):
    with path.open() as handle:
        return yaml.safe_load(handle)


rows = []
for path in sorted(SCENARIO_DIR.glob("*.yaml")):
    spec = load_yaml(path)
    rows.append(
        {
            "scenario_id": spec["id"],
            "difficulty": spec.get("difficulty"),
            "services": len(spec.get("topology", {}).get("services", [])),
            "incident_phases": len(spec.get("incident_chain", [])),
            "allowed_actions": len(spec.get("allowed_actions", [])),
            "reward_dimensions": len(spec.get("reward_dimensions", {})),
            "reference_trajectory_length": spec.get("reference_trajectory_length"),
            "max_ticks": spec.get("max_ticks"),
        }
    )

catalog = pd.DataFrame(rows).sort_values("scenario_id")
catalog

# %% [markdown]
# ## Cell 3 — Deep dive into one Strategy scenario

# %%
SCENARIO_ID = "cascading_release_train"
spec = load_yaml(SCENARIO_DIR / f"{SCENARIO_ID}.yaml")

print(f"Scenario: {spec['id']}")
print(f"Name    : {spec['name']}")
print(f"Tier    : {spec['tier']}")
print(f"\nDescription:\n{spec['description']}")

topology_df = pd.DataFrame(spec["topology"]["services"])
incident_df = pd.DataFrame(spec["incident_chain"])
reward_df = (
    pd.DataFrame(
        [{"dimension": key, **value} for key, value in spec["reward_dimensions"].items()]
    )
    .sort_values("dimension")
    .reset_index(drop=True)
)

basic_actions = {
    "query_logs",
    "query_metrics",
    "query_dependencies",
    "query_deploys",
    "rollback_deploy",
    "restart_service",
    "run_check",
    "isolate_service",
    "escalate",
    "submit_hypothesis",
    "declare_resolved",
}
strategy_actions = [action for action in spec["allowed_actions"] if action not in basic_actions]

print(f"\nTopology services: {len(topology_df)}")
display(topology_df)

print("\nIncident phases:")
display(incident_df)

print("\nReward dimensions:")
display(reward_df)

print(f"\nBasic actions carried over: {len(basic_actions)}")
print(f"Strategy-only documented actions: {len(strategy_actions)}")
print(strategy_actions)

# %% [markdown]
# ## Cell 4 — Reference trace and success criteria

# %%
phase_rows = []
for phase_name, steps in spec.get("reference_trace", {}).items():
    for step in steps:
        phase_rows.append(
            {
                "phase": phase_name,
                "tick": step.get("tick"),
                "action": step.get("action"),
                "expected_signal": step.get("expected_signal"),
            }
        )

trace_df = pd.DataFrame(phase_rows)
success_df = pd.DataFrame({"success_criteria": spec.get("success_criteria", [])})

print(f"Reference trace rows: {len(trace_df)}")
display(trace_df.head(20))

print("\nSuccess criteria:")
display(success_df)

print(
    "\nInterpretation: Strategy is still using the Triage 11-action runtime today, "
    "but these YAMLs define the horizon-level target state for future 7B-14B work."
)
