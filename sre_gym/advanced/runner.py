"""Deprecated alias for ``sre_gym.strategy.runner``."""
from __future__ import annotations
import warnings as _warnings
_warnings.warn(
    "sre_gym.advanced.runner is deprecated; import from sre_gym.strategy.runner",
    DeprecationWarning,
    stacklevel=2,
)
from sre_gym.strategy.runner import *  # noqa: F401,F403,E402
from sre_gym.strategy.runner import (  # noqa: F401,E402
    AdvancedResult,
    HORIZON_DECAY,
    HorizonState,
    PHASE_TO_BASIC_TEMPLATE,
    PhaseResult,
    list_advanced_scenarios,
    load_advanced_scenario,
    run_advanced,
)
