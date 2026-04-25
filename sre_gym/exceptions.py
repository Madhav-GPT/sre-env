"""Typed exceptions for sre-gym.

The cross-tier code uses these to surface structured errors to the UI / CLI
without leaking secrets or stack traces. All exceptions inherit from
``SREGymError`` so callers can do ``except SREGymError:`` for a catch-all.
"""

from __future__ import annotations


class SREGymError(Exception):
    """Base class for every sre-gym error."""


class TierUnavailableError(SREGymError):
    """Raised when a tier is requested that the current build cannot run.

    Carries a docs pointer so the UI can render a useful diagnostic.
    """

    def __init__(self, tier: str, message: str, *, docs_path: str = "") -> None:
        super().__init__(message)
        self.tier = tier
        self.docs_path = docs_path


class ScenarioLoadError(SREGymError):
    """Raised when a YAML / JSON scenario spec fails to load or validate."""

    def __init__(self, scenario_id: str, message: str) -> None:
        super().__init__(f"{scenario_id}: {message}")
        self.scenario_id = scenario_id


class ChaosPatternError(SREGymError):
    """Raised when a chaos pattern reference is invalid or unsafe to compose."""

    def __init__(self, pattern_id: str, message: str) -> None:
        super().__init__(f"{pattern_id}: {message}")
        self.pattern_id = pattern_id


class ProviderAuthError(SREGymError):
    """Raised when a provider call fails authentication.

    The caller MUST surface a redacted message — never include the failing key.
    """

    def __init__(self, provider: str, message: str = "auth failed") -> None:
        super().__init__(f"auth failed for provider '{provider}' — check your API key")
        self.provider = provider


class ProviderRateLimitError(SREGymError):
    """Raised when a provider call hits a 429 / rate-limit response."""

    def __init__(self, provider: str, retry_after_s: float | None = None) -> None:
        suffix = f" (retry in {retry_after_s:.1f}s)" if retry_after_s else ""
        super().__init__(f"provider '{provider}' rate-limited{suffix}")
        self.provider = provider
        self.retry_after_s = retry_after_s


class ProviderModelError(SREGymError):
    """Raised when a provider call fails for any non-auth, non-rate-limit reason."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"provider '{provider}': {message}")
        self.provider = provider


class ActionParseError(SREGymError):
    """Raised when the model output can't be parsed into a valid action."""

    def __init__(self, raw: str, reason: str) -> None:
        # Truncate raw to keep error logs small; never log secrets in raw.
        clipped = raw[:120] + ("…" if len(raw) > 120 else "")
        super().__init__(f"action parse failed ({reason}): {clipped}")
        self.reason = reason


class HorizonStateError(SREGymError):
    """Raised when an Advanced-tier horizon-state transition is impossible."""


class GraphSimulationError(SREGymError):
    """Raised when a Max-tier graph simulation hits an inconsistent state."""
