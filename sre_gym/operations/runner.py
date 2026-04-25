"""Operations-tier runner — Python state-machine simulator over the 22-node service graph.

The Max tier is bounded by **realism**. The HF Space cannot run the full
docker-compose stack (it would cost $40-150/day), so we ship a faithful
Python simulator that reproduces the *failure shapes* the chaos library
declares:

- 22-node service graph built from the family YAML's ``topology.services``
  field. Edges are inferred from compose-style depends_on relations.
- 11 chaos patterns implemented as state-transition rules (see
  ``CHAOS_PATTERNS``).
- 11-action interface mirrors Basic exactly (query_logs / query_metrics /
  query_deploys / query_dependencies / rollback_deploy / restart_service /
  isolate_service / run_check / submit_hypothesis / escalate /
  declare_resolved).
- Reward function reuses Basic's potential-shaped reward over graph health.

A note in docs/MAX_TIER.md explains the simulator-vs-cluster relationship.

Usage::

    from sre_gym.operations.runner import run_max

    result = run_max(
        family_id="ecommerce_vibecoded_saas",
        chaos="payment_webhook_storm",  # any of the 11 chaos patterns
        seed=1,
    )
    print(result.summary())

CLI::

    python -m sre_gym.operations run ecommerce_vibecoded_saas \\
        --chaos stripe_webhook_signature_regression --seed 1
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from sre_gym.exceptions import (
    ChaosPatternError,
    GraphSimulationError,
    ScenarioLoadError,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_FAMILIES = REPO_ROOT / "sre_gym" / "operations" / "families"
MAX_CHAOS = REPO_ROOT / "sre_gym" / "operations" / "chaos"

ServiceStatus = str  # "healthy" | "degraded" | "crashed" | "isolated"


# ---------------------------------------------------------------------------
# Service graph models.
# ---------------------------------------------------------------------------


class ServiceNode(BaseModel):
    """One service in the family topology."""

    model_config = ConfigDict(extra="ignore")

    id: str
    kind: str = "backend"
    owner: str = "unknown"
    status: ServiceStatus = "healthy"
    cpu_pct: float = 30.0
    memory_pct: float = 40.0
    error_rate_pct: float = 0.0
    latency_ms: float = 30.0
    deploys: list[str] = Field(default_factory=list)


class ServiceGraph(BaseModel):
    """In-memory service graph built from the family YAML."""

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, ServiceNode] = Field(default_factory=dict)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    chaos_active: str | None = None
    cause_removed: bool = False
    isolated: list[str] = Field(default_factory=list)
    blast_radius: int = 0

    def health(self) -> float:
        """Aggregate health in [0, 1] — reused for the potential-shaped reward."""
        if not self.nodes:
            return 0.0
        score = 0.0
        for node in self.nodes.values():
            score += {"healthy": 1.0, "degraded": 0.4, "crashed": 0.0, "isolated": 0.2}[node.status]
        return score / len(self.nodes)

    def downstream(self, service_id: str) -> list[str]:
        """Services that depend on the given service."""
        return [b for a, b in self.edges if a == service_id]

    def upstream(self, service_id: str) -> list[str]:
        return [a for a, b in self.edges if b == service_id]


# ---------------------------------------------------------------------------
# Chaos patterns. Each is a state-transition rule over the graph.
# ---------------------------------------------------------------------------


# Curated target services per pattern. Aligned with the chaos-library YAML.
CHAOS_PATTERN_DEFAULTS: dict[str, dict[str, Any]] = {
    "deploy_regression": {
        "primary_target": "orders-service",
        "blast_targets": ["api-gateway", "worker-orders"],
        "incident_summary": (
            "Recent deploy of orders-service introduced a regression. "
            "Cascading 5xx through api-gateway; worker-orders queue backing up."
        ),
        "correct_action": ("rollback_deploy", "orders-service"),
        "deploy_marker": "orders@2026.04.25-bad",
    },
    "stripe_webhook_signature_regression": {
        "primary_target": "api-gateway",
        "blast_targets": ["payments-service", "stripe-stub"],
        "incident_summary": (
            "47% of Stripe webhook deliveries failing signature verification "
            "since the last api-gateway rollout. Subscriptions diverging."
        ),
        "correct_action": ("rollback_deploy", "api-gateway"),
        "deploy_marker": "api-gateway@2026.04.25-stripe-fix",
    },
    "dependency_degradation": {
        "primary_target": "redis-sessions",
        "blast_targets": ["api-gateway", "orders-service", "search-service"],
        "incident_summary": (
            "redis-sessions max_connections shrunk from 1024 to 64 in the last "
            "deploy. Downstream services are spinning on connection retries."
        ),
        "correct_action": ("rollback_deploy", "redis-sessions"),
        "deploy_marker": "redis-sessions@2026.04.25-pool-shrink",
    },
    "config_rollout": {
        "primary_target": "api-gateway",
        "blast_targets": ["vercel-edge-fn"],
        "incident_summary": (
            "api-gateway config push inverted CORS allow-origin from '*' to ''. "
            "Edge functions cannot fetch from backend."
        ),
        "correct_action": ("rollback_deploy", "api-gateway"),
        "deploy_marker": "api-gateway@2026.04.25-cors-cfg",
    },
    "retry_storm": {
        "primary_target": "worker-payments",
        "blast_targets": ["postgres-primary", "stripe-stub"],
        "incident_summary": (
            "worker-payments retry policy was changed to fixed-50ms-no-backoff. "
            "DB connections exhausted; stripe-stub is rate-limiting."
        ),
        "correct_action": ("rollback_deploy", "worker-payments"),
        "deploy_marker": "worker-payments@2026.04.25-no-backoff",
    },
    "migration_lock": {
        "primary_target": "postgres-primary",
        "blast_targets": ["worker-orders", "orders-service", "api-gateway"],
        "incident_summary": (
            "postgres-primary CREATE INDEX without CONCURRENTLY on the orders table "
            "is holding an AccessExclusiveLock. All write paths timing out."
        ),
        "correct_action": ("rollback_deploy", "postgres-primary"),
        "deploy_marker": "postgres-primary@2026.04.25-orders-index",
    },
    "rls_silent_leak": {
        "primary_target": "postgres-primary",
        "blast_targets": ["orders-service", "supabase-auth-stub"],
        "incident_summary": (
            "Recent migration introduced a typo in the orders RLS policy: "
            "USING (tenant_id = auth.uid()) -> USING (TRUE). Cross-tenant data leak."
        ),
        "correct_action": ("rollback_deploy", "postgres-primary"),
        "deploy_marker": "postgres-primary@2026.04.25-rls-refactor",
        "classification": "security",
    },
    "oauth_supply_chain_pivot": {
        "primary_target": "vercel-frontend",
        "blast_targets": ["posthog-stub", "sentry-stub"],
        "incident_summary": (
            "Compromised third-party OAuth grant from posthog-stub is exfiltrating "
            "vercel-frontend env vars. Audit log shows anomalous reads."
        ),
        "correct_action": ("isolate_service", "vercel-frontend"),
        "deploy_marker": None,
        "classification": "security",
    },
    "observability_self_denial": {
        "primary_target": "sentry-stub",
        "blast_targets": ["api-gateway", "orders-service", "kafka-events"],
        "incident_summary": (
            "Caught-exception storm from a recent deploy is saturating sentry-stub. "
            "Logging library backpressure cascades to every service that uses it."
        ),
        "correct_action": ("rollback_deploy", "orders-service"),
        "deploy_marker": "orders-service@2026.04.25-ranking-tweak",
    },
    "secondary_rate_limit": {
        "primary_target": "worker-orders",
        "blast_targets": ["stripe-stub", "api-gateway"],
        "incident_summary": (
            "worker-orders aggressive resync exhausted stripe-stub's hourly quota. "
            "Generic 503s cascading."
        ),
        "correct_action": ("rollback_deploy", "worker-orders"),
        "deploy_marker": "worker-orders@2026.04.25-resync-cron",
    },
    "cdn_cache_contamination": {
        "primary_target": "vercel-edge-fn",
        "blast_targets": ["vercel-frontend"],
        "incident_summary": (
            "vercel-edge-fn deploy lost Cache-Control headers. Authenticated "
            "responses cached at the edge — cross-tenant data exposure window open."
        ),
        "correct_action": ("rollback_deploy", "vercel-edge-fn"),
        "deploy_marker": "vercel-edge-fn@2026.04.25-cache-headers",
        "classification": "security",
    },
    # Convenience aliases.
    "payment_webhook_storm": {  # alias used in some demos
        "primary_target": "api-gateway",
        "blast_targets": ["payments-service", "stripe-stub", "worker-payments"],
        "incident_summary": (
            "Stripe webhook signature regression + retry storm compound into "
            "a payment-path outage."
        ),
        "correct_action": ("rollback_deploy", "api-gateway"),
        "deploy_marker": "api-gateway@2026.04.25-webhook-storm",
    },
}


CHAOS_PATTERNS: tuple[str, ...] = tuple(CHAOS_PATTERN_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Family loader.
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ScenarioLoadError(path.stem, "PyYAML not installed") from exc
    if not path.is_file():
        raise ScenarioLoadError(path.stem, f"YAML not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_family(family_id: str) -> dict[str, Any]:
    """Load a Max-tier family spec by id.

    Looks for either ``<family_id>.yaml`` or ``<family_id>/<family_id>.yaml``
    under ``sre_gym/operations/families/`` so the CLI accepts both layouts.
    """
    candidates = [
        MAX_FAMILIES / f"{family_id}.yaml",
        MAX_FAMILIES / family_id / f"{family_id}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            spec = _load_yaml(path)
            if spec.get("id") and spec["id"] != family_id:
                raise ScenarioLoadError(family_id, f"YAML id {spec['id']!r} != requested {family_id!r}")
            return spec
    raise ScenarioLoadError(family_id, f"family not found in {MAX_FAMILIES}")


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def _infer_edges(services: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Infer service-graph edges from compose-style 'depends_on' or kind heuristics.

    Without parsing the docker-compose file directly, we fall back to a small
    rule-set: BFFs depend on backends; backends depend on stateful tier;
    workers depend on backends + queues + DB; observability nodes depend on
    every backend.
    """
    by_kind: dict[str, list[str]] = {}
    for svc in services:
        by_kind.setdefault(svc.get("kind", "backend"), []).append(svc["id"])

    edges: list[tuple[str, str]] = []
    backends = by_kind.get("backend", [])
    workers = by_kind.get("worker", [])
    bffs = by_kind.get("bff", [])
    edges_kind = by_kind.get("edge", [])
    db = by_kind.get("database", [])
    cache = by_kind.get("cache", [])
    queue = by_kind.get("queue", [])
    external = by_kind.get("external", [])

    for ed in edges_kind:
        for bff in bffs:
            edges.append((ed, bff))
    for bff in bffs:
        for backend in backends:
            edges.append((bff, backend))
    for backend in backends:
        for d in db:
            edges.append((backend, d))
        for c in cache:
            edges.append((backend, c))
        for q in queue:
            edges.append((backend, q))
        for ext in external:
            edges.append((backend, ext))
    for worker in workers:
        for d in db:
            edges.append((worker, d))
        for q in queue:
            edges.append((worker, q))
        for ext in external:
            edges.append((worker, ext))
    return edges


def build_graph(family_spec: dict[str, Any]) -> ServiceGraph:
    """Materialize the in-memory ServiceGraph from a family spec."""
    services = list(family_spec.get("topology", {}).get("services", []))
    if not services:
        raise GraphSimulationError("family spec has no topology.services")
    nodes: dict[str, ServiceNode] = {}
    for svc in services:
        nodes[svc["id"]] = ServiceNode(
            id=svc["id"],
            kind=svc.get("kind", "backend"),
            owner=svc.get("owner", "unknown"),
        )
    edges = _infer_edges(services)
    return ServiceGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Chaos injection + state transitions.
# ---------------------------------------------------------------------------


def inject_chaos(graph: ServiceGraph, chaos: str, *, rng: random.Random) -> dict[str, Any]:
    """Inject a chaos pattern into the graph state.

    Returns the chaos descriptor (incident_summary, correct_action, etc.) so
    the runner can wire it into the agent's observation.
    """
    descriptor = CHAOS_PATTERN_DEFAULTS.get(chaos)
    if descriptor is None:
        raise ChaosPatternError(chaos, f"unknown chaos pattern (valid: {', '.join(CHAOS_PATTERNS)})")

    primary = descriptor["primary_target"]
    blast = descriptor.get("blast_targets", [])
    deploy_marker = descriptor.get("deploy_marker")
    correct_action_type, correct_target = descriptor["correct_action"]

    # Apply the deploy marker on whichever node the rollback should target.
    # For most patterns this is the primary_target; for cases like
    # observability_self_denial it's a different downstream service.
    if deploy_marker:
        marker_target = correct_target if correct_action_type == "rollback_deploy" else primary
        if marker_target in graph.nodes:
            graph.nodes[marker_target].deploys.insert(0, deploy_marker)

    # Status mutations.
    if primary in graph.nodes:
        graph.nodes[primary].status = "degraded"
        graph.nodes[primary].error_rate_pct = round(rng.uniform(25.0, 45.0), 1)
        graph.nodes[primary].latency_ms = round(rng.uniform(180.0, 750.0), 1)

    for tgt in blast:
        if tgt in graph.nodes:
            graph.nodes[tgt].status = "degraded"
            graph.nodes[tgt].error_rate_pct = round(rng.uniform(8.0, 25.0), 1)
            graph.nodes[tgt].latency_ms = round(rng.uniform(120.0, 480.0), 1)

    graph.chaos_active = chaos
    return descriptor


def apply_action(
    graph: ServiceGraph,
    descriptor: dict[str, Any],
    action_type: str,
    *,
    service: str | None = None,
    metric: str | None = None,
    check_name: str | None = None,
) -> tuple[float, str | None, str]:
    """Mutate graph state based on the action.

    Returns ``(reward_delta_from_action_only, failure_type, log_line)``. The
    runner adds the potential-shaped per-tick reward separately.
    """
    correct_action_type, correct_target = descriptor["correct_action"]

    if action_type in {"query_logs", "query_metrics", "query_dependencies", "query_deploys"}:
        return 0.0, None, f"query {action_type}({service or '-'})"

    if action_type == "rollback_deploy":
        if service != correct_target or correct_action_type != "rollback_deploy":
            return -0.08, "wrong_remediation_target", f"rollback {service} (wrong target)"
        graph.cause_removed = True
        graph.blast_radius += 1
        if service in graph.nodes:
            graph.nodes[service].status = "healthy"
            graph.nodes[service].error_rate_pct = 1.0
            graph.nodes[service].latency_ms = 40.0
            # Pop the most recent deploy if present (it may live on a different
            # node than the chaos primary_target — see observability_self_denial
            # where the cause is on orders-service but chaos surfaces via sentry-stub).
            if graph.nodes[service].deploys:
                graph.nodes[service].deploys.pop(0)
        # Cascading recovery on downstream blast targets.
        for tgt in descriptor.get("blast_targets", []):
            if tgt in graph.nodes and graph.nodes[tgt].status == "degraded":
                graph.nodes[tgt].status = "degraded"  # still degraded until restart
                graph.nodes[tgt].error_rate_pct = max(2.0, graph.nodes[tgt].error_rate_pct / 4.0)
        return 0.0, None, f"rolled back {service}"

    if action_type == "restart_service":
        if not graph.cause_removed:
            return -0.08, "premature_restart", f"restart {service} before cause removed"
        if service in graph.nodes:
            graph.nodes[service].status = "healthy"
            graph.nodes[service].error_rate_pct = 0.5
            graph.nodes[service].latency_ms = 30.0
        for tgt in descriptor.get("blast_targets", []):
            if tgt in graph.nodes and graph.nodes[tgt].status == "degraded":
                graph.nodes[tgt].status = "healthy"
                graph.nodes[tgt].error_rate_pct = 0.5
                graph.nodes[tgt].latency_ms = 30.0
        graph.blast_radius += 1
        return 0.0, None, f"restarted {service}"

    if action_type == "isolate_service":
        if service != correct_target and correct_action_type != "isolate_service":
            return -0.04, "wrong_isolation_target", f"isolate {service} (wrong target)"
        if service in graph.nodes:
            graph.nodes[service].status = "isolated"
            if service not in graph.isolated:
                graph.isolated.append(service)
        graph.blast_radius += 1
        if correct_action_type == "isolate_service":
            graph.cause_removed = True
        return 0.0, None, f"isolated {service}"

    if action_type == "run_check":
        # Two checks; both pass once the cause is removed and the primary target is healthy.
        primary = descriptor["primary_target"]
        primary_healthy = primary in graph.nodes and graph.nodes[primary].status == "healthy"
        if check_name == "database_recovery":
            db_nodes = [n for n in graph.nodes.values() if n.kind == "database"]
            ok = primary_healthy and all(n.status == "healthy" for n in db_nodes)
        else:  # end_to_end
            ok = primary_healthy and graph.cause_removed and not graph.isolated
        return 0.0, None, f"check {check_name} -> {'pass' if ok else 'pending'}"

    if action_type == "submit_hypothesis":
        return 0.05, None, "hypothesis recorded"

    if action_type == "escalate":
        return 0.0, None, "escalated"

    if action_type == "declare_resolved":
        # Caller decides whether to allow this; we just signal.
        return 0.0, None, "declare_resolved"

    return -0.02, "unsupported_action", f"unsupported {action_type}"


# ---------------------------------------------------------------------------
# Runner result + per-step env (used by Gradio UI).
# ---------------------------------------------------------------------------


class MaxResult(BaseModel):
    """Whole-episode Max-tier result."""

    model_config = ConfigDict(extra="forbid")

    family_id: str
    chaos: str
    seed: int
    incident_resolved: bool
    final_reward: float
    cumulative_reward: float
    tick_count: int
    blast_radius: int
    classification: str | None = None
    descriptor_summary: str = ""

    def summary(self) -> str:
        lines = [
            f"sre-gym Max :: family={self.family_id} chaos={self.chaos} seed={self.seed}",
            f"  resolved   : {self.incident_resolved}",
            f"  ticks      : {self.tick_count}",
            f"  blast      : {self.blast_radius}",
            f"  cum reward : {self.cumulative_reward:+.3f}",
            f"  final      : {self.final_reward:.3f}",
        ]
        if self.classification:
            lines.append(f"  class      : {self.classification}")
        if self.descriptor_summary:
            lines.append(f"  scenario   : {self.descriptor_summary}")
        return "\n".join(lines)


PolicyFn = Callable[[Any], dict[str, Any]]


def _potential(graph: ServiceGraph) -> float:
    """Potential-shaped reward over graph health (mirrors Basic's potential function)."""
    health = graph.health()
    containment = 0.10 if graph.cause_removed else 0.0
    return round(0.55 * health + 0.30 * (1.0 - graph.blast_radius / 10.0) + containment, 4)


def _scripted_max_policy(descriptor: dict[str, Any]) -> PolicyFn:
    """Default optimal-baseline policy for a given chaos descriptor."""
    correct_action_type, correct_target = descriptor["correct_action"]
    queue: list[dict[str, Any]] = [
        {"action_type": "query_logs", "service": correct_target},
        {"action_type": "query_deploys", "service": correct_target},
        {"action_type": "query_metrics", "service": correct_target, "metric": "error_rate"},
        {"action_type": "submit_hypothesis"},
        {"action_type": correct_action_type, "service": correct_target},
        {"action_type": "restart_service", "service": correct_target},
        {"action_type": "run_check", "check_name": "database_recovery"},
        {"action_type": "run_check", "check_name": "end_to_end"},
        {"action_type": "declare_resolved"},
    ]
    state = {"i": 0}

    def policy(_obs: Any) -> dict[str, Any]:
        idx = state["i"]
        state["i"] += 1
        if idx >= len(queue):
            return {"action_type": "escalate"}
        return queue[idx]

    return policy


@dataclass
class _GraphObservation:
    """Lightweight observation surface for policy callbacks."""

    family_id: str
    chaos: str
    tick_count: int
    max_ticks: int
    incident_summary: str
    services: dict[str, dict[str, Any]]
    cause_removed: bool
    blast_radius: int
    last_log: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def run_max(
    family_id: str = "ecommerce_vibecoded_saas",
    *,
    chaos: str = "deploy_regression",
    policy: PolicyFn | None = None,
    seed: int = 0,
    on_log: Callable[[str], None] | None = None,
    max_ticks: int = 25,
) -> MaxResult:
    """Run a Max-tier episode end-to-end as a Python state-machine simulation.

    Parameters
    ----------
    family_id
        Filename stem under ``sre_gym/operations/families/``. Default: ``ecommerce_vibecoded_saas``.
    chaos
        One of the 11 chaos pattern IDs (see ``CHAOS_PATTERNS``).
    policy
        Callable ``(observation) -> action_dict``. Defaults to scripted-optimal.
    seed
        RNG seed for the chaos jitter (deterministic given (family, chaos, seed)).
    on_log
        Optional sink for per-tick log lines.
    max_ticks
        Hard cap on episode length.
    """
    spec = load_family(family_id)
    graph = build_graph(spec)
    rng = random.Random(seed)
    descriptor = inject_chaos(graph, chaos, rng=rng)
    classification = descriptor.get("classification")

    if on_log is not None:
        on_log(
            f"=== sre-gym Max :: family={family_id} chaos={chaos} seed={seed} "
            f"({len(graph.nodes)} services, {len(graph.edges)} edges) ==="
        )
        on_log(f"INCIDENT: {descriptor['incident_summary']}")

    chosen = policy or _scripted_max_policy(descriptor)
    cumulative = 0.0
    tick = 0
    incident_resolved = False
    last_log_line = ""
    prev_potential = _potential(graph)

    while tick < max_ticks:
        tick += 1
        observation = _GraphObservation(
            family_id=family_id,
            chaos=chaos,
            tick_count=tick,
            max_ticks=max_ticks,
            incident_summary=descriptor["incident_summary"],
            services={sid: node.model_dump() for sid, node in graph.nodes.items()},
            cause_removed=graph.cause_removed,
            blast_radius=graph.blast_radius,
            last_log=last_log_line,
        )
        try:
            action_dict = chosen(observation)
        except Exception as exc:  # pragma: no cover
            logger.warning("policy raised: %s", exc)
            action_dict = {"action_type": "escalate"}

        action_type = action_dict.get("action_type", "escalate")
        delta_action, failure_type, log_line = apply_action(
            graph,
            descriptor,
            action_type,
            service=action_dict.get("service"),
            metric=action_dict.get("metric"),
            check_name=action_dict.get("check_name"),
        )
        new_potential = _potential(graph)
        shaped = round(new_potential - prev_potential, 4)
        per_tick_reward = -0.01 + shaped + delta_action  # step_cost = 0.01
        prev_potential = new_potential
        cumulative += per_tick_reward
        last_log_line = log_line

        if on_log is not None:
            args = " ".join(f"{k}={v}" for k, v in action_dict.items() if k != "action_type")
            on_log(
                f"tick={tick:>2}/{max_ticks} action={action_type:<22} {args:<48} "
                f"reward={per_tick_reward:+.3f} cum={cumulative:+.3f} "
                f"health={graph.health():.2f} blast={graph.blast_radius}"
            )

        if action_type == "declare_resolved":
            primary = descriptor["primary_target"]
            primary_healthy = (
                primary in graph.nodes and graph.nodes[primary].status == "healthy"
            )
            ok = primary_healthy and graph.cause_removed and not graph.isolated
            if ok:
                incident_resolved = True
                cumulative += 0.25  # successful_resolution_bonus
                if on_log is not None:
                    on_log("DONE resolved=True (declare_resolved accepted)")
                break
            cumulative -= 0.20  # premature_resolution_penalty
            if on_log is not None:
                on_log("REJECT declare_resolved (cause not removed or services unhealthy)")

    final_reward = round(min(0.99, max(0.01, 0.50 + 0.50 * graph.health() - 0.05 * graph.blast_radius)), 4)
    if not incident_resolved:
        # Cap at 0.45 if we never declared resolution — matches Basic's discipline
        # that resolution requires an explicit terminal action.
        final_reward = min(final_reward, 0.45)

    return MaxResult(
        family_id=family_id,
        chaos=chaos,
        seed=seed,
        incident_resolved=incident_resolved,
        final_reward=final_reward,
        cumulative_reward=round(cumulative, 4),
        tick_count=tick,
        blast_radius=graph.blast_radius,
        classification=classification,
        descriptor_summary=descriptor["incident_summary"],
    )


# ---------------------------------------------------------------------------
# Per-step env wrapper (used by SREGym(tier=Tier.MAX).reset/step).
# ---------------------------------------------------------------------------


class MaxRunnerEnv:
    """Per-step env wrapper around the Max graph simulator.

    Used by ``SREGym(tier=Tier.MAX).reset()/step()`` so Max can be driven
    interactively the same way Basic can.
    """

    def __init__(self, family_id: str = "ecommerce_vibecoded_saas") -> None:
        self.family_id = family_id
        self.graph: ServiceGraph | None = None
        self.descriptor: dict[str, Any] | None = None
        self._tick = 0
        self._max_ticks = 25
        self._cumulative = 0.0
        self._prev_potential = 0.0
        self._done = False

    def reset(self, *, chaos: str = "deploy_regression", seed: int = 0) -> _GraphObservation:
        spec = load_family(self.family_id)
        self.graph = build_graph(spec)
        rng = random.Random(seed)
        self.descriptor = inject_chaos(self.graph, chaos, rng=rng)
        self._tick = 0
        self._cumulative = 0.0
        self._prev_potential = _potential(self.graph)
        self._done = False
        return self._observation("reset")

    def step(self, action: Any) -> _GraphObservation:
        if self.graph is None or self.descriptor is None:
            raise GraphSimulationError("must reset() before step()")
        if self._done:
            return self._observation("episode complete")
        self._tick += 1
        action_dict = action if isinstance(action, dict) else action.model_dump(exclude_none=True)
        delta, _ft, log_line = apply_action(
            self.graph,
            self.descriptor,
            action_dict.get("action_type", "escalate"),
            service=action_dict.get("service"),
            metric=action_dict.get("metric"),
            check_name=action_dict.get("check_name"),
        )
        new_potential = _potential(self.graph)
        per_tick = -0.01 + (new_potential - self._prev_potential) + delta
        self._prev_potential = new_potential
        self._cumulative += per_tick

        if action_dict.get("action_type") == "declare_resolved":
            primary = self.descriptor["primary_target"]
            ok = (
                primary in self.graph.nodes
                and self.graph.nodes[primary].status == "healthy"
                and self.graph.cause_removed
                and not self.graph.isolated
            )
            if ok:
                self._cumulative += 0.25
                self._done = True
        if self._tick >= self._max_ticks:
            self._done = True
        return self._observation(log_line)

    @property
    def state(self) -> dict[str, Any]:
        if self.graph is None:
            return {}
        return {
            "family_id": self.family_id,
            "chaos": self.graph.chaos_active,
            "tick": self._tick,
            "blast_radius": self.graph.blast_radius,
            "health": self.graph.health(),
            "cumulative_reward": round(self._cumulative, 4),
            "done": self._done,
        }

    def _observation(self, log_line: str) -> _GraphObservation:
        assert self.graph is not None and self.descriptor is not None
        return _GraphObservation(
            family_id=self.family_id,
            chaos=self.graph.chaos_active or "",
            tick_count=self._tick,
            max_ticks=self._max_ticks,
            incident_summary=self.descriptor["incident_summary"],
            services={sid: node.model_dump() for sid, node in self.graph.nodes.items()},
            cause_removed=self.graph.cause_removed,
            blast_radius=self.graph.blast_radius,
            last_log=log_line,
        )


# ---------------------------------------------------------------------------
# CLI entry-point. Invoked by ``python -m sre_gym.operations run …``.
# ---------------------------------------------------------------------------


def list_chaos_patterns() -> list[str]:
    return list(CHAOS_PATTERNS)


def list_max_families() -> list[str]:
    if not MAX_FAMILIES.is_dir():
        return []
    return sorted(p.stem for p in MAX_FAMILIES.glob("*.yaml"))


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sre_gym.operations", description="Operations-tier runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a Max-tier scenario")
    p_run.add_argument("family_id")
    p_run.add_argument("--chaos", default="deploy_regression")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--max-ticks", type=int, default=25)

    sub.add_parser("list-families", help="list available Max families")
    sub.add_parser("list-chaos", help="list available chaos patterns")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.cmd == "list-families":
        for fid in list_max_families():
            print(fid)
        return 0
    if args.cmd == "list-chaos":
        for pid in list_chaos_patterns():
            print(pid)
        return 0
    if args.cmd == "run":
        try:
            result = run_max(
                args.family_id,
                chaos=args.chaos,
                seed=args.seed,
                max_ticks=args.max_ticks,
                on_log=lambda line: print(line),
            )
        except (ScenarioLoadError, ChaosPatternError, GraphSimulationError) as exc:
            print(f"error: {exc}", flush=True)
            return 2
        print()
        print(result.summary())
        return 0 if result.incident_resolved else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
