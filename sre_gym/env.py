"""Tier-aware environment factory.

``SREGym`` is the public-facing entry point.  Constructing it with
``tier=Tier.BASIC`` returns a runnable environment that delegates every method
call to ``unified_incident_env.UnifiedIncidentEnvironment`` — i.e. the existing
Hugging Face Space surface.  The Advanced and Max tiers raise a structured
``TierNotRunnableError`` carrying a pointer to the design doc and any data
artifacts shipped for that tier (reference scenarios, family specs, compose
files).

This indirection is the difference between "a single-tier env that's hard to
extend" and "an env that visibly carries the three-tier story even if only one
tier is trained against".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .tier import TIER_CONFIGS, Tier, TierConfig


REPO_ROOT = Path(__file__).resolve().parent.parent


class TierNotRunnableError(NotImplementedError):
    """Raised when a non-runnable tier is invoked end-to-end.

    The exception carries the tier config + design-doc path so the caller can
    surface the right pointer.  This is *not* an error in the testing sense —
    Advanced and Max are deliberately design-only in this repo.
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

    # ------- Runnable surface -------
    def reset(self, **kwargs: Any) -> Any:
        if self._delegate is None:
            raise TierNotRunnableError(
                self.tier,
                f"Tier {self.tier.value} is design-only in this repo. "
                f"See {self.config.docs_path} for the spec; the Basic tier is the "
                f"runnable surface (clone the repo and use Tier.BASIC).",
                docs_path=self.config.docs_path,
            )
        return self._delegate.reset(**kwargs)

    def step(self, action: Any, **kwargs: Any) -> Any:
        if self._delegate is None:
            raise TierNotRunnableError(
                self.tier,
                f"Tier {self.tier.value} step() not implemented.",
                docs_path=self.config.docs_path,
            )
        return self._delegate.step(action, **kwargs)

    @property
    def state(self) -> Any:
        if self._delegate is None:
            raise TierNotRunnableError(
                self.tier,
                f"Tier {self.tier.value} state not implemented.",
                docs_path=self.config.docs_path,
            )
        return self._delegate.state

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
