"""
Expert trajectory collection — Claude acting as teacher model.
Uses the Python environment API directly (not HTTP) for stateful episode execution.
Encodes the optimal action sequence for each of the 6 missing templates.
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

from unified_incident_env.models import UnifiedIncidentAction, HypothesisPayload
from unified_incident_env.server.challenge import get_scenario
from unified_incident_env.server.environment import UnifiedIncidentEnvironment

EXPERT_THRESHOLD = 0.85


def _hypo(root_cause: str, affected_services: list, confidence: float = 0.9,
           next_action: str = "rollback_deploy") -> UnifiedIncidentAction:
    return UnifiedIncidentAction(
        action_type="submit_hypothesis",
        hypothesis=HypothesisPayload(
            root_cause=root_cause,
            affected_services=affected_services,
            confidence=confidence,
            recommended_next_action=next_action,
        ),
    )


def _qlog(service: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="query_logs", service=service)


def _qdep(service: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="query_deploys", service=service)


def _qmet(service: str, metric: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="query_metrics", service=service, metric=metric)


def _qdeps(service: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="query_dependencies", service=service)


def _rollback(service: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="rollback_deploy", service=service)


def _restart(service: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="restart_service", service=service)


def _check(name: str) -> UnifiedIncidentAction:
    return UnifiedIncidentAction(action_type="run_check", check_name=name)


_DECLARE = UnifiedIncidentAction(action_type="declare_resolved")


# ── optimal plans per template ────────────────────────────────────────────────
PLANS: dict[str, list[UnifiedIncidentAction]] = {
    "auth_token_expiry": [
        _qdep("worker"),
        _qlog("worker"),
        _qmet("worker", "error_rate"),
        _qmet("api-gateway", "error_rate"),
        _hypo("credential_rotation_breakage", ["worker", "api-gateway"]),
        _rollback("worker"),
        _restart("worker"),
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
    "dep_degradation": [
        _qdep("cache"),
        _qlog("worker"),
        _qmet("worker", "error_rate"),
        _qdeps("worker"),
        _hypo("dependency_pool_exhausted", ["cache", "worker", "api-gateway"], 0.85),
        _rollback("cache"),
        _restart("cache"),
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
    "memory_leak_oom": [
        _qdep("worker"),
        _qmet("worker", "cpu"),
        _qlog("worker"),
        _qmet("database", "error_rate"),
        _hypo("memory_leak_runaway", ["worker", "database", "api-gateway"], 0.85),
        _rollback("worker"),
        _restart("database"),   # drain DB connection backlog after OOM restarts
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
    "migration_lock": [
        _qdep("database"),
        _qlog("database"),
        _qmet("database", "latency"),
        _qmet("worker", "error_rate"),
        _hypo("migration_lock_contention", ["database", "worker", "api-gateway"], 0.88),
        _rollback("database"),
        _restart("database"),
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
    "network_partition": [
        _qdep("cache"),
        _qdeps("worker"),
        _qlog("cache"),
        _qmet("worker", "error_rate"),
        _hypo("network_dns_partition", ["cache", "worker", "api-gateway"], 0.85),
        _rollback("cache"),
        _restart("cache"),
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
    "rate_limit_retry_storm": [
        _qdep("worker"),
        _qlog("worker"),
        _qmet("worker", "error_rate"),
        _qmet("database", "latency"),
        _hypo("external_rate_limit_storm", ["worker", "database", "api-gateway"], 0.85),
        _rollback("worker"),
        _restart("database"),   # drain open-transaction backlog
        _check("database_recovery"),
        _check("end_to_end"),
        _DECLARE,
    ],
}

MISSING_TEMPLATES = list(PLANS.keys())


def collect_episode(*, scenario_id: str, plan_key: str, attempt: int) -> dict:
    """Run one episode against the Python env using the hard-coded expert plan."""
    actions = PLANS[plan_key]
    scenario = get_scenario(scenario_id)

    env = UnifiedIncidentEnvironment()
    env._episode = env._make_episode(scenario)

    trajectory: list[dict] = []
    start = time.perf_counter()

    obs = None
    for i, action in enumerate(actions):
        prompt_text = env._prompt_text(tool_output=None)

        obs = env.step(action)
        reward = float(obs.reward) if hasattr(obs, 'reward') else 0.0

        response_text = json.dumps(
            {k: v for k, v in action.model_dump(exclude_none=True).items()
             if k not in ("hypothesis",)} |
            ({"hypothesis": action.hypothesis.model_dump()} if action.hypothesis else {}),
            separators=(",", ":"),
        )

        trajectory.append({
            "tick": int(obs.tick_count),
            "prompt": prompt_text,
            "response_text": response_text,
            "action": action.model_dump(exclude_none=True),
            "reward": reward,
            "tool_output": obs.tool_output,
            "failure_type": obs.failure_type,
            "workflow_stage": obs.workflow_stage,
        })

        score = float(obs.final_score) if hasattr(obs, 'final_score') else 0.0
        print(f"    tick {i+1:>2}  {action.action_type:<26}  "
              f"reward={reward:.3f}  score={score:.3f}  stage={obs.workflow_stage}"
              + (f"  FAIL:{obs.failure_type}" if obs.failure_type else ""))

        if getattr(obs, 'done', False):
            break

    elapsed = time.perf_counter() - start
    final_score = float(obs.final_score) if obs else 0.0
    resolved = bool(obs.incident_resolved) if obs else False

    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario_id,
        "template_id": plan_key,
        "model": "claude-expert-scripted",
        "driver": "scripted",
        "seed": attempt,
        "difficulty": scenario.get("difficulty", "medium"),
        "final_score": final_score,
        "incident_resolved": resolved,
        "steps": int(obs.tick_count) if obs else len(trajectory),
        "elapsed_s": round(elapsed, 3),
        "score_breakdown": dict(obs.score_breakdown) if obs and obs.score_breakdown else {},
        "trajectory": trajectory,
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_batch": f"expert_scripted_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "quality_tier": "expert" if final_score >= EXPERT_THRESHOLD else
                         ("mediocre" if final_score >= 0.30 else "failure"),
    }


def main() -> int:
    output = REPO_ROOT / "train" / "data" / "sonnet_missing6.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes_per_template = 5  # base + __p01..__p04
    written = 0
    failed: list[str] = []

    with output.open("w") as f:
        for tid in MISSING_TEMPLATES:
            print(f"\n=== {tid} ===")
            scenarios = [tid] + [f"{tid}__p0{i}" for i in range(1, episodes_per_template)]
            for attempt, sid in enumerate(scenarios[:episodes_per_template]):
                print(f"  {sid} ...")
                try:
                    episode = collect_episode(scenario_id=sid, plan_key=tid, attempt=attempt)
                    f.write(json.dumps(episode) + "\n")
                    f.flush()
                    written += 1
                    print(f"    => score={episode['final_score']:.3f}  "
                          f"resolved={episode['incident_resolved']}  "
                          f"tier={episode['quality_tier']}  steps={episode['steps']}")
                except Exception as exc:
                    print(f"    => ERROR: {exc}")
                    failed.append(sid)

    print(f"\nWrote {written} episodes to {output}")
    if failed:
        print(f"Failed scenarios: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
