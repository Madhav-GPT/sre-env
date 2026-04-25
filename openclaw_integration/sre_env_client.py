"""Deprecation shim — use ``coliseum.client`` instead."""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "openclaw_integration.sre_env_client is deprecated; "
    "import from `coliseum.client` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from coliseum.client import (  # noqa: E402,F401
    ArenaClient as SreEnvClient,
    create_arena_client as create_env_client,
)

__all__ = ["SreEnvClient", "create_env_client"]
