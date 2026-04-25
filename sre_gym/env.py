"""Tier-aware environment factory.

``SREGym`` is the public-facing entry point.  Constructing it with
``tier=Tier.BASIC`` returns a runnable environment that delegates every method
call to ``unified_incident_env.UnifiedIncidentEnvironment`` — i.e. the existing
Hugging Face Space surface.  Advanced and Max are also runnable now via the
``run()`` method, which dispatches to ``sre_gym.advanced.runner.run_advanced``
(chained Basic episodes with horizon state) and ``sre_gym.max.runner.run_max``
(Python state-machine simulator over the 22-node service graph).

The legacy ``reset() / step() / state`` per-step interface remains supported
for Basic. Advanced is *episodic* (run a multi-phase scenario end-to-end)
rather than per-step, so its ``reset()`` raises with a pointer to ``run()``.
Max also supports per-step via ``MaxRunnerEnv`` (returned by ``run()`` when
the caller passes ``stream=True``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from .exceptions import TierUnavailableError
from .tier import TIER_CONFIGS, Tier, TierConfig


REPO_ROOT = Path(__file__).resolve().parent.parent


class TierNotRunnableError(NotImplementedError):
    """Legacy alias kept for backward compatibility.

    The Advanced and Max tiers are now runnable via ``SREGym.run()``. This
    exception still raises if you call the per-step ``reset() / step() / state``
    interface on a tier that only supports episodic execution (Advanced).

    The exception carries the tier config + design-doc path so the caller can
    surface the right pointer.
    """

    def __init__(self, tier: Tier, message: str, *, docs_path: str = "") -> None:
        super().__init__(message)
        self.tier = tier
        self.docs_path = docs_path


class SREGym:
    """Tier-aware factory.

    Examples
    --------
    >>> env = SREGym(tier=Tier.BASIC)
    >>> obs = env.reset(scenario_id="memory_leak_oom__p02")
    >>> obs = env.step({"action_type": "rollback_deploy", "service": "worker"})

    For the non-runnable tiers you can still introspect the design space:

    >>> env = SREGym(tier=Tier.ADVANCED)
    >>> for spec in env.list_scenarios():
    ...     print(spec["id"], spec["multi_incident_chain"])

    But calling reset/step on a non-runnable tier raises ``TierNotRunnableError``.
    """

    def __init__(self, tier: Tier | str = Tier.BASIC) -> None:
        if isinstance(tier, str):
            tier = Tier(tier)
        self.tier: Tier = tier
        self.config: TierConfig = TIER_CONFIGS[tier]
        self._delegate: Any = None
        if tier is Tier.BASIC:
            from unified_incident_env.server.environment import UnifiedIncidentEnvironment
            self._delegate = UnifiedIncidentEnvironment()

    # ------- Runnable surface (Basic per-step, all tiers via run()) -------
    def reset(self, **kwargs: Any) -> Any:
        if self.tier is Tier.BASIC and self._delegate is not None:
            return self._delegate.reset(**kwargs)
        if self.tier is Tier.MAX:
            from sre_gym.max.runner import MaxRunnerEnv

            self._delegate = MaxRunnerEnv(family_id=kwargs.get("family_id", "ecommerce_vibecoded_saas"))
            chaos = kwargs.get("chaos") or "deploy_regression"
            seed = int(kwargs.get("seed", 0))
            return self._delegate.reset(chaos=chaos, seed=seed)
        raise TierNotRunnableError(
            self.tier,
            f"Tier {self.tier.value} is episodic — use SREGym(tier).run(scenario_id) "
            f"or `python -m sre_gym.{self.tier.value} run …` instead of per-step reset().",
            docs_path=self.config.docs_path,
        )

    def step(self, action: Any, **kwargs: Any) -> Any:
        if self._delegate is None:
            raise TierNotRunnableError(
                self.tier,
                f"Tier {self.tier.value} requires reset() before step().",
                docs_path=self.config.docs_path,
            )
        return self._delegate.step(action, **kwargs)

    @property
    def state(self) -> Any:
        if self._delegate is None:
            raise TierNotRunnableError(
                self.tier,
                f"Tier {self.tier.value} requires reset() before reading state.",
                docs_path=self.config.docs_path,
            )
        return self._delegate.state

    def run(
        self,
        scenario_id: str,
        *,
        policy: Callable[[Any], dict[str, Any]] | None = None,
        seed: int = 0,
        on_log: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a scenario end-to-end on the active tier.

        Returns
        -------
        For Basic, returns a Pydantic ``BasicResult`` (single-episode rollout).
        For Advanced, returns ``AdvancedResult`` (multi-phase chained episodes
        with horizon-decay).
        For Max, returns ``MaxResult`` (graph-state trace + outcome score).
        """
        if self.tier is Tier.BASIC:
            from sre_gym.basic_runner import run_basic

            return run_basic(scenario_id, policy=policy, seed=seed, on_log=on_log)
        if self.tier is Tier.ADVANCED:
            from sre_gym.advanced.runner import run_advanced

            return run_advanced(scenario_id, policy=policy, seed=seed, on_log=on_log)
        if self.tier is Tier.MAX:
            from sre_gym.max.runner import run_max

            chaos = kwargs.get("chaos") or "deploy_regression"
            return run_max(scenario_id, chaos=chaos, policy=policy, seed=seed, on_log=on_log)
        raise TierUnavailableError(self.tier.value, f"unknown tier {self.tier!r}")

    # ------- Introspection surface (works on all tiers) -------
    def describe(self) -> dict[str, Any]:
        """Return the tier's escalation dimension, persona, compute budget, etc."""
        cfg = self.config
        return {
            "tier": cfg.tier.value,
            "escalation_dimension": cfg.escalation_dimension,
            "persona": cfg.persona,
            "compute_budget": cfg.expected_compute_budget,
            "scenario_count": cfg.scenario_count,
            "scenario_template_count": cfg.scenario_template_count,
            "procgen_variants_per_template": cfg.procgen_variants_per_template,
            "action_count": cfg.expected_action_count,
            "max_episode_ticks": cfg.max_episode_ticks,
            "observation_richness": cfg.observation_richness,
            "runnable_in_repo": cfg.runnable,
            "docs": cfg.docs_path,
            "notes": cfg.notes,
        }

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List the scenario specs available for this tier.

        For the runnable Basic tier this is the live procgen catalogue; for the
        data-only tiers it's the YAML/JSON specs in their respective directories.
        """
        if self.tier is Tier.BASIC:
            from unified_incident_env.server.challenge import list_scenarios

            return [s.model_dump() for s in list_scenarios().scenarios]

        glob = self.config.scenarios_glob
        if not glob:
            return []
        return list(_iter_yaml_specs(REPO_ROOT, glob))


def _iter_yaml_specs(root: Path, glob: str) -> Iterable[dict[str, Any]]:
    """Read all YAML scenario specs under root matching *glob*.

    PyYAML is loaded lazily so importing the package on machines without it
    (e.g. judges' CI runners that only need Basic) doesn't fail.
    """
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - optional dep
        return
    for path in sorted(root.glob(glob)):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict):
            data.setdefault("_source", str(path.relative_to(root)))
            yield data
