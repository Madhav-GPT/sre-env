"""Deprecated alias for ``sre_gym.strategy``.

This module re-exports everything from ``sre_gym.strategy`` so legacy
``import sre_gym.advanced`` callers keep working. Removal: v3.2.
"""
from __future__ import annotations
import warnings as _warnings

_warnings.warn(
    "sre_gym.advanced is deprecated; import from sre_gym.strategy",
    DeprecationWarning,
    stacklevel=2,
)
from sre_gym.strategy import *  # noqa: F401,F403,E402
