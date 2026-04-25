"""Deprecated alias for ``sre_gym.operations.runner``."""
from __future__ import annotations
import warnings as _warnings
_warnings.warn(
    "sre_gym.max.runner is deprecated; import from sre_gym.operations.runner",
    DeprecationWarning,
    stacklevel=2,
)
from sre_gym.operations.runner import *  # noqa: F401,F403,E402
from sre_gym.operations.runner import (  # noqa: F401,E402
    CHAOS_PATTERNS,
    CHAOS_PATTERN_DEFAULTS,
    MaxResult,
    MaxRunnerEnv,
    ServiceGraph,
    ServiceNode,
    list_max_families,
    load_family,
    run_max,
)
