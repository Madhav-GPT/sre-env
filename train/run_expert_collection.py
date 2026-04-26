"""
Diverse expert trajectory collection.

Generates 4 plans per template × 5 procgen variants = 20 expert trajectories
per template × 6 sonnet templates = 120 expert episodes.

Each plan has a distinct evidence-gathering ordering, which is what prevents
SFT memorization (the policy collapse we hit on the homogeneous v1 data).

Run with:
    python train/run_expert_collection.py
Output:
    train/data/sonnet_diverse.jsonl      (120 expert episodes)
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from unified_incident_env.models import HypothesisPayload, UnifiedIncidentAction
from unified_incident_env.server.challenge import get_scenario
from unified_incident_env.server.environment import UnifiedIncidentEnvironment

EXPERT_THRESHOLD = 0.85


# ── action factories ──────────────────────────────────────────────────────────
def _qlog(s):     return UnifiedIncidentAction(action_type="query_logs", service=s)
def _qdep(s):     return UnifiedIncidentAction(action_type="query_deploys", service=s)
def _qmet(s, m):  return UnifiedIncidentAction(action_type="query_metrics", service=s, metric=m)
def _qdeps(s):    return UnifiedIncidentAction(action_type="query_dependencies", service=s)
def _rb(s):       return UnifiedIncidentAction(action_type="rollback_deploy", service=s)
def _rs(s):       return UnifiedIncidentAction(action_type="restart_service", service=s)
def _ck(name):    return UnifiedIncidentAction(action_type="run_check", check_name=name)
_DECLARE = UnifiedIncidentAction(action_type="declare_resolved")


def _hypo(root_cause, affected, conf, next_action="rollback_deploy"):
    return UnifiedIncidentAction(
        action_type="submit_hypothesis",
        hypothesis=HypothesisPayload(
            root_cause=root_cause,
            affected_services=affected,
            confidence=conf,
            recommended_next_action=next_action,
        ),
    )


# ── 4 distinct plans per template, varying query order/types ──────────────────
def _plans_for(template: str):
    """Return 4 distinct plans (lists of actions) for the given template.

    Each plan has the same hypothesis/rollback/restart/checks core but
    different evidence-gathering preludes and slightly different hypothesis
    confidence values.
    """
    if template == "auth_token_expiry":
        rb, rs = "worker", "worker"
        rc = "credential_rotation_breakage"
        affected = ["worker", "api-gateway"]
        evidence_plans = [
            [_qdep("worker"),  _qlog("worker"),  _qmet("worker","error_rate"), _qmet("api-gateway","error_rate")],
            [_qlog("worker"),  _qdep("worker"),  _qmet("api-gateway","error_rate"), _qmet("worker","latency")],
            [_qmet("worker","error_rate"), _qdep("worker"), _qdeps("worker"), _qlog("worker")],
            [_qdep("worker"),  _qmet("worker","latency"), _qlog("worker")],
        ]

    elif template == "dep_degradation":
        rb, rs = "cache", "cache"
        rc = "dependency_pool_exhausted"
        affected = ["cache", "worker", "api-gateway"]
        evidence_plans = [
            [_qdep("cache"),   _qlog("worker"),  _qmet("worker","error_rate"), _qdeps("worker")],
            [_qdeps("worker"), _qdep("cache"),   _qlog("worker"),               _qmet("cache","latency")],
            [_qlog("worker"),  _qmet("worker","error_rate"), _qdep("cache"),    _qdeps("worker")],
            [_qdep("cache"),   _qdeps("worker"), _qlog("cache")],
        ]

    elif template == "memory_leak_oom":
        rb, rs = "worker", "database"
        rc = "memory_leak_runaway"
        affected = ["worker", "database", "api-gateway"]
        evidence_plans = [
            [_qdep("worker"),  _qmet("worker","cpu"), _qlog("worker"), _qmet("database","error_rate")],
            [_qmet("worker","cpu"), _qdep("worker"), _qmet("database","latency"), _qlog("worker")],
            [_qlog("worker"),  _qdep("worker"),  _qmet("worker","cpu"),         _qmet("database","error_rate")],
            [_qdep("worker"),  _qmet("worker","cpu"), _qmet("database","error_rate")],
        ]

    elif template == "migration_lock":
        rb, rs = "database", "database"
        rc = "migration_lock_contention"
        affected = ["database", "worker", "api-gateway"]
        evidence_plans = [
            [_qdep("database"), _qlog("database"),  _qmet("database","latency"), _qmet("worker","error_rate")],
            [_qlog("database"), _qmet("database","latency"), _qdep("database"),  _qmet("worker","error_rate")],
            [_qmet("database","latency"), _qdep("database"), _qlog("database"),  _qmet("worker","latency")],
            [_qdep("database"), _qmet("database","latency"), _qlog("database")],
        ]

    elif template == "network_partition":
        rb, rs = "cache", "cache"
        rc = "network_dns_partition"
        affected = ["cache", "worker", "api-gateway"]
        evidence_plans = [
            [_qdep("cache"),   _qdeps("worker"), _qlog("cache"), _qmet("worker","error_rate")],
            [_qdeps("worker"), _qdep("cache"),   _qmet("worker","error_rate"), _qlog("cache")],
            [_qlog("cache"),   _qdep("cache"),   _qdeps("worker"), _qmet("worker","latency")],
            [_qdep("cache"),   _qdeps("worker"), _qlog("worker")],
        ]

    elif template == "rate_limit_retry_storm":
        rb, rs = "worker", "database"
        rc = "external_rate_limit_storm"
        affected = ["worker", "database", "api-gateway"]
        evidence_plans = [
            [_qdep("worker"),  _qlog("worker"),  _qmet("worker","error_rate"), _qmet("database","latency")],
            [_qmet("worker","error_rate"), _qdep("worker"), _qlog("worker"), _qmet("database","error_rate")],
            [_qlog("worker"),  _qmet("worker","error_rate"), _qdep("worker"), _qdeps("worker")],
            [_qdep("worker"),  _qmet("worker","error_rate"), _qlog("worker")],
        ]

    elif template == "worker_deploy_cascade":
        rb, rs = "worker", "database"
        rc = "bad_worker_deploy"
        affected = ["worker", "database", "api-gateway"]
        evidence_plans = [
            [_qdep("worker"),  _qlog("worker"),  _qmet("worker","error_rate"), _qmet("database","latency")],
            [_qlog("worker"),  _qdep("worker"),  _qmet("database","error_rate"), _qmet("worker","cpu")],
            [_qmet("worker","error_rate"), _qlog("worker"), _qdep("worker"), _qmet("database","latency")],
            [_qdep("worker"),  _qmet("database","error_rate"), _qlog("database")],
        ]

    elif template == "db_config_rollout":
        rb, rs = "database", "database"
        rc = "database_only_failure"
        affected = ["database", "api-gateway", "worker"]
        evidence_plans = [
            [_qdep("database"), _qlog("database"),  _qmet("database","error_rate"), _qmet("worker","error_rate")],
            [_qmet("database","latency"), _qdep("database"), _qlog("database"),     _qmet("database","error_rate")],
            [_qlog("database"), _qmet("database","error_rate"), _qdep("database"),  _qmet("worker","latency")],
            [_qdep("database"), _qlog("database"), _qmet("database","error_rate")],
        ]

    elif template == "gateway_auth_rollout":
        # api-gateway rollback only — no restart for this template
        rb, rs = "api-gateway", None
        rc = "api_gateway_fault"
        affected = ["api-gateway", "worker"]
        evidence_plans = [
            [_qdep("api-gateway"), _qlog("api-gateway"), _qmet("api-gateway","error_rate"), _qmet("worker","error_rate")],
            [_qlog("api-gateway"), _qdep("api-gateway"), _qmet("worker","error_rate"),       _qmet("api-gateway","latency")],
            [_qmet("api-gateway","error_rate"), _qdep("api-gateway"), _qlog("api-gateway"), _qmet("worker","latency")],
            [_qdep("api-gateway"), _qlog("api-gateway"), _qmet("api-gateway","error_rate")],
        ]

    elif template == "payment_webhook_misconfig":
        rb, rs = "api-gateway", None
        rc = "payment_webhook_regression"
        affected = ["api-gateway", "database"]
        evidence_plans = [
            [_qdep("api-gateway"), _qlog("api-gateway"), _qmet("api-gateway","error_rate"), _qmet("database","latency")],
            [_qlog("api-gateway"), _qdep("api-gateway"), _qmet("database","error_rate"),     _qmet("api-gateway","latency")],
            [_qmet("api-gateway","error_rate"), _qlog("api-gateway"), _qdep("api-gateway"), _qmet("database","latency")],
            [_qdep("api-gateway"), _qmet("api-gateway","error_rate"), _qlog("database")],
        ]

    elif template == "schema_drift_missing_migration":
        rb, rs = "api-gateway", None
        rc = "schema_migration_mismatch"
        affected = ["api-gateway", "worker", "database"]
        evidence_plans = [
            [_qdep("api-gateway"), _qlog("api-gateway"), _qmet("api-gateway","error_rate"), _qmet("worker","error_rate")],
            [_qlog("api-gateway"), _qdep("api-gateway"), _qmet("worker","error_rate"),       _qmet("database","error_rate")],
            [_qmet("api-gateway","error_rate"), _qdep("api-gateway"), _qlog("api-gateway"), _qlog("worker")],
            [_qdep("api-gateway"), _qlog("worker"), _qmet("database","error_rate")],
        ]

    elif template == "cache_stale_state":
        rb, rs = "cache", "cache"
        rc = "cache_ttl_regression"
        affected = ["cache", "api-gateway"]
        evidence_plans = [
            [_qdep("cache"),   _qlog("cache"),   _qmet("cache","latency"),       _qmet("api-gateway","error_rate")],
            [_qlog("cache"),   _qdep("cache"),   _qmet("api-gateway","latency"), _qmet("cache","error_rate")],
            [_qmet("cache","latency"), _qdep("cache"), _qdeps("api-gateway"),    _qlog("cache")],
            [_qdep("cache"),   _qmet("api-gateway","latency"), _qlog("cache")],
        ]

    else:
        raise ValueError(f"No plans defined for {template}")

    # Vary hypothesis confidence per plan to add response diversity at the
    # token level (the hypothesis JSON is part of the assistant response).
    confidences = [0.85, 0.88, 0.92, 0.95]

    plans = []
    for i, evidence in enumerate(evidence_plans):
        plan = list(evidence) + [_hypo(rc, affected, confidences[i]), _rb(rb)]
        if rs is not None:                           # some templates need no restart
            plan.append(_rs(rs))
        plan.extend([
            _ck("database_recovery"),
            _ck("end_to_end"),
            _DECLARE,
        ])
        plans.append(plan)
    return plans


ALL_TEMPLATES = [
    # 6 sonnet (round-2) templates
    "auth_token_expiry", "dep_degradation", "memory_leak_oom",
    "migration_lock", "network_partition", "rate_limit_retry_storm",
    # 6 round-1 templates
    "worker_deploy_cascade", "db_config_rollout", "gateway_auth_rollout",
    "payment_webhook_misconfig", "schema_drift_missing_migration", "cache_stale_state",
]


def collect_episode(*, scenario_id: str, plan_actions, plan_idx: int, attempt: int) -> dict:
    """Run one episode against the Python env using the given plan."""
    scenario = get_scenario(scenario_id)
    env = UnifiedIncidentEnvironment()
    env._episode = env._make_episode(scenario)

    trajectory = []
    start = time.perf_counter()
    obs = None

    for i, action in enumerate(plan_actions):
        prompt_text = env._prompt_text(tool_output=None)
        obs = env.step(action)
        reward = float(obs.reward) if hasattr(obs, "reward") else 0.0

        # Serialize action to JSON (the model will be trained to produce this)
        action_dict = action.model_dump(exclude_none=True)
        if action.hypothesis is not None:
            action_dict["hypothesis"] = action.hypothesis.model_dump()
        response_text = json.dumps(action_dict, separators=(",", ":"))

        trajectory.append({
            "tick": int(obs.tick_count),
            "prompt": prompt_text,
            "response_text": response_text,
            "action": action_dict,
            "reward": reward,
            "tool_output": obs.tool_output,
            "failure_type": obs.failure_type,
            "workflow_stage": obs.workflow_stage,
        })

        if getattr(obs, "done", False):
            break

    elapsed = time.perf_counter() - start
    final_score = float(obs.final_score) if obs else 0.0
    resolved = bool(obs.incident_resolved) if obs else False

    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario_id,
        "template_id": scenario_id.split("__")[0],
        "model": "claude-expert-scripted",
        "driver": "scripted-diverse",
        "seed": plan_idx * 100 + attempt,
        "difficulty": scenario.get("difficulty", "medium"),
        "final_score": final_score,
        "incident_resolved": resolved,
        "steps": int(obs.tick_count) if obs else len(trajectory),
        "elapsed_s": round(elapsed, 3),
        "score_breakdown": dict(obs.score_breakdown) if obs and obs.score_breakdown else {},
        "trajectory": trajectory,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_batch": f"diverse_v2_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "plan_index": plan_idx,
        "quality_tier": "expert" if final_score >= EXPERT_THRESHOLD else
                        ("mediocre" if final_score >= 0.30 else "failure"),
    }


def main() -> int:
    output = REPO_ROOT / "train" / "data" / "sonnet_diverse.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = []
    summary = {}

    with output.open("w") as f:
        for tid in ALL_TEMPLATES:
            print(f"\n=== {tid} ===")
            plans = _plans_for(tid)
            scenarios = [tid] + [f"{tid}__p0{i}" for i in range(1, 5)]   # 5 procgen variants

            for plan_idx, plan in enumerate(plans):
                for attempt, sid in enumerate(scenarios):
                    try:
                        ep = collect_episode(
                            scenario_id=sid, plan_actions=plan,
                            plan_idx=plan_idx, attempt=attempt,
                        )
                    except Exception as exc:
                        print(f"  [FAIL] plan={plan_idx} sid={sid}: {exc}")
                        failed.append((tid, plan_idx, sid))
                        continue
                    f.write(json.dumps(ep) + "\n")
                    f.flush()
                    written += 1
                    if plan_idx == 0 and attempt == 0:
                        print(f"  plan {plan_idx} on {sid:<35} score={ep['final_score']:.3f} tier={ep['quality_tier']} steps={ep['steps']}")

            summary[tid] = sum(1 for _ in plans) * len(scenarios)

    print(f"\nWrote {written} episodes to {output}")
    print(f"Per template: {summary}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]}{'...' if len(failed)>5 else ''}")

    # Quick uniqueness check — how many distinct response sequences?
    by_template = {}
    with output.open() as f:
        for line in f:
            ep = json.loads(line)
            tid = ep["template_id"]
            seq = tuple(t["response_text"] for t in ep["trajectory"])
            by_template.setdefault(tid, set()).add(seq)
    print("\nUnique response sequences per template:")
    for tid, seqs in by_template.items():
        print(f"  {tid:<28} {len(seqs)} unique")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
