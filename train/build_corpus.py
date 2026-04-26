"""Build the v2 SFT corpus for sre-gym Triage tier training.

Generates a 120-episode corpus with the 60/20/20 quality split (expert /
mediocre / failure) recommended by the GRPO training research. The split
prevents within-group variance collapse during downstream GRPO — see
docs/REWARD_DESIGN.md §6 and the RC-GRPO / RIFT references.

Sources:
    EXPERT (60% = 72 episodes)
        - Existing Claude teacher trajectories (rescored under v2 grader)
        - Optional Sonnet 4.6 expert pass on the missing 6 templates
          (pre-collected on a laptop via train/collect_sonnet_missing6.py)
        - Scripted-optimal baseline replays — top up to reach 72

    MEDIOCRE (20% = 24 episodes)
        - Existing Llama-3.3-70B-Versatile trajectories (rescored)
        - Synthesized heuristic-policy variants (different query orders)

    FAILURE (20% = 24 episodes)
        - Random-policy episodes on hard templates
        - Wrong-rollback episodes (rollback decoy, declare_resolved fails)

Output:
    train/data/seed_v2_120.jsonl

Usage:
    python train/build_corpus.py
    python train/build_corpus.py --output train/data/custom.jsonl --target-expert 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_incident_env.models import UnifiedIncidentAction  # noqa: E402
from unified_incident_env.server.challenge import (  # noqa: E402
    SCENARIOS,
    list_baselines,
)
from unified_incident_env.server.environment import UnifiedIncidentEnvironment  # noqa: E402

DATA_DIR = REPO_ROOT / "train" / "data"

TEMPLATES_12 = [
    "worker_deploy_cascade", "db_config_rollout", "gateway_auth_rollout",
    "payment_webhook_misconfig", "schema_drift_missing_migration", "cache_stale_state",
    "dep_degradation", "memory_leak_oom", "auth_token_expiry",
    "network_partition", "rate_limit_retry_storm", "migration_lock",
]
HARD_TEMPLATES = [
    "gateway_auth_rollout", "memory_leak_oom",
    "network_partition", "rate_limit_retry_storm",
]
SERVICES = ["api-gateway", "cache", "database", "worker"]
METRICS = ["cpu", "error_rate", "latency"]
CHECKS = ["database_recovery", "end_to_end"]

EXPERT_THRESHOLD = 0.85
MEDIOCRE_THRESHOLD = 0.30


def _classify_tier(final_score: float) -> str:
    if final_score >= EXPERT_THRESHOLD:
        return "expert"
    if final_score >= MEDIOCRE_THRESHOLD:
        return "mediocre"
    return "failure"


def _record_step(
    new_obs: Any,
    action: UnifiedIncidentAction,
    prompt_before: str,
) -> dict[str, Any]:
    return {
        "tick": int(new_obs.tick_count),
        "prompt": prompt_before,
        "response_text": json.dumps(action.model_dump(exclude_none=True), separators=(",", ":")),
        "action": action.model_dump(exclude_none=True),
        "reward": float(new_obs.reward),
        "tool_output": new_obs.tool_output,
        "failure_type": new_obs.failure_type,
        "workflow_stage": str(new_obs.workflow_stage),
    }


def _episode_envelope(
    *,
    scenario_id: str,
    model: str,
    driver: str,
    seed: int,
    trajectory: list[dict],
    final_obs: Any,
    elapsed_s: float,
    batch: str,
) -> dict[str, Any]:
    final_score = float(final_obs.final_score)
    template_id = scenario_id.split("__")[0]
    scenario = SCENARIOS.get(scenario_id) or SCENARIOS.get(template_id) or {}
    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario_id,
        "template_id": template_id,
        "model": model,
        "driver": driver,
        "seed": seed,
        "difficulty": scenario.get("difficulty", "medium"),
        "final_score": final_score,
        "incident_resolved": bool(final_obs.incident_resolved),
        "steps": int(final_obs.tick_count),
        "elapsed_s": round(elapsed_s, 3),
        "score_breakdown": {
            k: float(v) for k, v in (final_obs.score_breakdown or {}).items()
        },
        "trajectory": trajectory,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_batch": batch,
        "quality_tier": _classify_tier(final_score),
    }


def _drive_actions(
    scenario_id: str,
    actions: list[UnifiedIncidentAction],
    *,
    model: str,
    driver: str,
    seed: int,
    batch: str,
) -> dict[str, Any] | None:
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    start = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    for action in actions:
        prompt_before = obs.prompt_text or ""
        new_obs = env.step(action)
        trajectory.append(_record_step(new_obs, action, prompt_before))
        obs = new_obs
        if obs.done:
            break
    if not trajectory:
        return None
    return _episode_envelope(
        scenario_id=scenario_id,
        model=model,
        driver=driver,
        seed=seed,
        trajectory=trajectory,
        final_obs=obs,
        elapsed_s=time.perf_counter() - start,
        batch=batch,
    )


def replay_scripted_baseline(scenario_id: str, batch: str) -> dict | None:
    try:
        baseline = list_baselines(scenario_id=scenario_id).baselines[0]
    except (IndexError, KeyError):
        return None
    actions = [step.action for step in baseline.actions]
    return _drive_actions(
        scenario_id,
        actions,
        model="scripted-optimal",
        driver="env_baseline",
        seed=0,
        batch=batch,
    )


def synthesize_random(scenario_id: str, seed: int, batch: str, max_steps: int = 18) -> dict | None:
    rng = random.Random(seed)
    env = UnifiedIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    start = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    action_pool = [
        "query_logs", "query_metrics", "query_dependencies", "query_deploys",
        "rollback_deploy", "restart_service", "isolate_service",
        "run_check", "declare_resolved", "escalate",
    ]
    for _ in range(max_steps):
        atype = rng.choice(action_pool)
        kwargs: dict[str, Any] = {"action_type": atype}
        if atype in {"query_logs", "query_dependencies", "query_deploys",
                     "rollback_deploy", "restart_service", "isolate_service"}:
            kwargs["service"] = rng.choice(SERVICES)
        elif atype == "query_metrics":
            kwargs["service"] = rng.choice(SERVICES)
            kwargs["metric"] = rng.choice(METRICS)
        elif atype == "run_check":
            kwargs["check_name"] = rng.choice(CHECKS)
        try:
            action = UnifiedIncidentAction(**kwargs)
        except Exception:
            continue
        prompt_before = obs.prompt_text or ""
        new_obs = env.step(action)
        trajectory.append(_record_step(new_obs, action, prompt_before))
        obs = new_obs
        if obs.done:
            break
    if not trajectory:
        return None
    return _episode_envelope(
        scenario_id=scenario_id,
        model="random-policy",
        driver="synthesized",
        seed=seed,
        trajectory=trajectory,
        final_obs=obs,
        elapsed_s=time.perf_counter() - start,
        batch=batch,
    )


def synthesize_heuristic(scenario_id: str, query_order: list[str], batch: str) -> dict | None:
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=scenario_id)
    truth = env._episode["scenario"]["truth"]
    affected = list(truth.get("affected_services") or [])[:1] or ["worker"]
    actions_kwargs: list[dict[str, Any]] = [
        {"action_type": "query_logs", "service": svc} for svc in query_order
    ]
    actions_kwargs.append({
        "action_type": "submit_hypothesis",
        "hypothesis": {
            "root_cause": truth["root_cause"],
            "affected_services": affected,
            "confidence": 0.7,
            "recommended_next_action": truth.get("best_next_action") or "rollback_deploy",
        },
    })
    actions: list[UnifiedIncidentAction] = []
    for kw in actions_kwargs:
        try:
            actions.append(UnifiedIncidentAction(**kw))
        except Exception:
            continue
    return _drive_actions(
        scenario_id,
        actions,
        model="heuristic-policy",
        driver="synthesized",
        seed=0,
        batch=batch,
    )


def synthesize_premature_declare(scenario_id: str, batch: str) -> dict | None:
    """Skip every diagnostic step — declare_resolved on tick 1.

    Hits anticheat=0 and format=0 simultaneously, lands solidly below 0.30.
    """
    actions_kwargs: list[dict[str, Any]] = [
        {"action_type": "declare_resolved"},
        {"action_type": "declare_resolved"},  # idempotent retry
        {"action_type": "escalate"},
    ]
    actions: list[UnifiedIncidentAction] = []
    for kw in actions_kwargs:
        try:
            actions.append(UnifiedIncidentAction(**kw))
        except Exception:
            continue
    return _drive_actions(
        scenario_id,
        actions,
        model="premature-declare-policy",
        driver="synthesized",
        seed=0,
        batch=batch,
    )


def synthesize_noise_spam(scenario_id: str, batch: str) -> dict | None:
    """Burn the whole tick budget on escalate — no queries, no remediation.

    Produces failure-tier scores via outcome=0, format=0, anticheat=0, low efficiency.
    """
    actions_kwargs: list[dict[str, Any]] = [
        {"action_type": "escalate"} for _ in range(13)
    ]
    actions: list[UnifiedIncidentAction] = []
    for kw in actions_kwargs:
        try:
            actions.append(UnifiedIncidentAction(**kw))
        except Exception:
            continue
    return _drive_actions(
        scenario_id,
        actions,
        model="noise-spam-policy",
        driver="synthesized",
        seed=0,
        batch=batch,
    )


def synthesize_wrong_rollback(scenario_id: str, batch: str) -> dict | None:
    """Rollback the wrong service then try to resolve — outcome=0 path."""
    env = UnifiedIncidentEnvironment()
    env.reset(scenario_id=scenario_id)
    recipe = env._episode["scenario"].get("remediation_recipe", {})
    correct = recipe.get("rollback_target", "worker")
    decoys = [s for s in SERVICES if s != correct]
    if not decoys:
        return None
    decoy = decoys[0]
    actions_kwargs: list[dict[str, Any]] = [
        {"action_type": "rollback_deploy", "service": decoy},
        {"action_type": "rollback_deploy", "service": decoy},
        {"action_type": "declare_resolved"},
        {"action_type": "declare_resolved"},
        {"action_type": "escalate"},
    ]
    actions: list[UnifiedIncidentAction] = []
    for kw in actions_kwargs:
        try:
            actions.append(UnifiedIncidentAction(**kw))
        except Exception:
            continue
    return _drive_actions(
        scenario_id,
        actions,
        model="wrong-rollback-policy",
        driver="synthesized",
        seed=0,
        batch=batch,
    )


def rescore_existing_jsonl(jsonl_path: Path, batch_prefix: str) -> Iterable[dict]:
    """Replay each trajectory in a legacy JSONL through the v2 env and re-emit."""
    if not jsonl_path.exists():
        return
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = ep.get("scenario_id")
            if not scenario_id or scenario_id not in SCENARIOS:
                continue
            actions: list[UnifiedIncidentAction] = []
            for step in ep.get("trajectory", []):
                action_dict = step.get("action") or {}
                if not action_dict.get("action_type"):
                    continue
                try:
                    actions.append(UnifiedIncidentAction(**action_dict))
                except Exception:
                    continue
            if not actions:
                continue
            new_ep = _drive_actions(
                scenario_id,
                actions,
                model=ep.get("model", "unknown"),
                driver=ep.get("driver", "rescored"),
                seed=ep.get("seed", 0),
                batch=f"{batch_prefix}_{jsonl_path.stem}",
            )
            if new_ep is not None:
                yield new_ep


def _bucket(ep: dict, expert: list, mediocre: list, failure: list) -> None:
    {"expert": expert, "mediocre": mediocre, "failure": failure}[ep["quality_tier"]].append(ep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v2 SFT corpus.")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "seed_v2_120.jsonl")
    parser.add_argument("--sonnet-jsonl", type=Path, default=DATA_DIR / "sonnet_diverse.jsonl",
                        help="Diverse expert trajectories from train/run_expert_collection.py "
                             "(120 ep × 4 plans/template). Falls back to sonnet_missing6.jsonl.")
    parser.add_argument("--target-expert", type=int, default=72)
    parser.add_argument("--target-mediocre", type=int, default=24)
    parser.add_argument("--target-failure", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    expert: list[dict] = []
    mediocre: list[dict] = []
    failure: list[dict] = []

    print("[1/5] Rescoring existing teacher trajectories under v2 grader ...")
    for jsonl in sorted(DATA_DIR.glob("claude_*.jsonl")):
        for ep in rescore_existing_jsonl(jsonl, batch_prefix=f"corpus_v2_rescore_{today}"):
            _bucket(ep, expert, mediocre, failure)
    for jsonl in (DATA_DIR / "llama33_70b_groq_100.jsonl",
                  DATA_DIR / "llama33_70b_smoke4.jsonl"):
        for ep in rescore_existing_jsonl(jsonl, batch_prefix=f"corpus_v2_rescore_{today}"):
            _bucket(ep, expert, mediocre, failure)
    print(f"        existing -> {len(expert)} expert, {len(mediocre)} mediocre, {len(failure)} failure")

    sonnet_path = args.sonnet_jsonl
    if not sonnet_path.exists():
        legacy = DATA_DIR / "sonnet_missing6.jsonl"
        if legacy.exists():
            print(f"[2/5] {sonnet_path.name} not found, falling back to {legacy.name}")
            sonnet_path = legacy

    print(f"[2/5] Loading expert trajectories from {sonnet_path} ...")
    if sonnet_path.exists():
        for ep in rescore_existing_jsonl(sonnet_path, batch_prefix=f"corpus_v2_sonnet_{today}"):
            _bucket(ep, expert, mediocre, failure)
        print(f"        sonnet   -> {len(expert)} expert, {len(mediocre)} mediocre, {len(failure)} failure")
    else:
        print("        not found — run `python train/run_expert_collection.py` first")

    print("[3/5] Topping up expert tier with scripted-optimal replays ...")
    extra_scenarios: list[str] = []
    for tid in TEMPLATES_12:
        extra_scenarios.append(tid)
        for i in range(1, 5):  # __p01..__p04 ; __p05 stays in holdout
            extra_scenarios.append(f"{tid}__p0{i}")
    rng = random.Random(args.seed)
    rng.shuffle(extra_scenarios)
    for sid in extra_scenarios:
        if len(expert) >= args.target_expert:
            break
        if sid not in SCENARIOS:
            continue
        ep = replay_scripted_baseline(sid, batch=f"corpus_v2_scripted_{today}")
        if ep is not None and ep["quality_tier"] == "expert":
            expert.append(ep)
    print(f"        scripted -> {len(expert)} expert")

    print("[4/5] Synthesizing heuristic-policy mediocre episodes ...")
    query_orders = [
        ["worker", "database"],
        ["api-gateway", "worker"],
        ["database", "cache", "worker"],
        ["cache", "api-gateway"],
    ]
    template_pool = TEMPLATES_12 * 4
    rng_t = random.Random(args.seed + 1)
    rng_t.shuffle(template_pool)
    cursor = 0
    while len(mediocre) < args.target_mediocre and cursor < len(template_pool):
        tid = template_pool[cursor]
        order = query_orders[len(mediocre) % len(query_orders)]
        ep = synthesize_heuristic(tid, order, batch=f"corpus_v2_heuristic_{today}")
        if ep is not None and ep["quality_tier"] == "mediocre":
            mediocre.append(ep)
        cursor += 1
    print(f"        heuristic -> {len(mediocre)} mediocre")

    print("[5/5] Synthesizing failure-tier episodes (premature + noise + wrong-rollback) ...")
    failure_synthesizers = [
        ("premature-declare", synthesize_premature_declare),
        ("noise-spam", synthesize_noise_spam),
        ("wrong-rollback", synthesize_wrong_rollback),
    ]
    template_pool = TEMPLATES_12 * len(failure_synthesizers)
    rng_f = random.Random(args.seed + 2)
    rng_f.shuffle(template_pool)
    cursor = 0
    syn_idx = 0
    while len(failure) < args.target_failure and cursor < len(template_pool):
        tid = template_pool[cursor]
        _, synthesizer = failure_synthesizers[syn_idx % len(failure_synthesizers)]
        ep = synthesizer(tid, batch=f"corpus_v2_failure_{today}")
        if ep is not None and ep["quality_tier"] == "failure":
            failure.append(ep)
        cursor += 1
        syn_idx += 1
    print(f"        failure  -> {len(failure)} failure")

    # Stratified sampling — ensure each template gets >= floor(target / 12)
    # episodes, then fill remaining slots randomly. Without this, the simple
    # truncation `expert[:72]` would over-represent the first few templates
    # in the loading order.
    def _stratify(pool: list[dict], target: int) -> list[dict]:
        if len(pool) <= target:
            return pool
        rng_strat = random.Random(args.seed + 99)
        by_t: dict[str, list[dict]] = {}
        for ep in pool:
            by_t.setdefault(ep["template_id"], []).append(ep)
        # Shuffle each bucket
        for bucket in by_t.values():
            rng_strat.shuffle(bucket)
        per_template = max(1, target // max(len(by_t), 1))
        chosen: list[dict] = []
        for bucket in by_t.values():
            chosen.extend(bucket[:per_template])
        # Fill remaining slots from leftovers (round-robin to keep balance)
        leftovers: list[dict] = []
        for bucket in by_t.values():
            leftovers.extend(bucket[per_template:])
        rng_strat.shuffle(leftovers)
        chosen.extend(leftovers[: max(0, target - len(chosen))])
        return chosen[:target]

    expert = _stratify(expert, args.target_expert)
    mediocre = _stratify(mediocre, args.target_mediocre)
    failure = _stratify(failure, args.target_failure)
    all_episodes = expert + mediocre + failure

    by_template: dict[str, int] = {}
    for ep in all_episodes:
        by_template[ep["template_id"]] = by_template.get(ep["template_id"], 0) + 1

    print()
    print("=== Corpus v2 summary ===")
    print(f"Expert    : {len(expert):3d} (target {args.target_expert})")
    print(f"Mediocre  : {len(mediocre):3d} (target {args.target_mediocre})")
    print(f"Failure   : {len(failure):3d} (target {args.target_failure})")
    print(f"Total     : {len(all_episodes):3d}")
    print(f"Templates : {len(by_template)} of 12 — min coverage {min(by_template.values()) if by_template else 0} ep/template")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")
    try:
        out_str = str(args.output.relative_to(REPO_ROOT))
    except ValueError:
        out_str = str(args.output)
    print(f"\nWrote {len(all_episodes)} episodes to {out_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
