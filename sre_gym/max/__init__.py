"""Deprecated alias for ``sre_gym.operations``.

Removal: v3.2.
"""
from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "sre_gym.max is deprecated; import from sre_gym.operations",
    DeprecationWarning,
    stacklevel=2,
)
from sre_gym.operations import *  # noqa: F401,F403,E402
