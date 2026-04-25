"""Deprecation shim — use ``coliseum.server`` instead."""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "openclaw_integration.pool_server is deprecated; "
    "import from `coliseum.server` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from coliseum.server import ArenaPool as LeasePool, app  # noqa: E402,F401

__all__ = ["LeasePool", "app"]
