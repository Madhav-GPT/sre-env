"""Deprecation shim — ``openclaw_integration`` is now ``coliseum``.

The package was renamed to remove the third-party trainer's name from the
public surface. The lease-based contract is unchanged; only the module
name, class names, and ``COLISEUM_*`` env vars differ.

Migration:

    # Old                                       # New
    from openclaw_integration import \\        from coliseum import \\
        SreEnvClient, LeasePool                     ArenaClient, ArenaPool

    POOL_SERVER_LEASE_TTL_S=600                 COLISEUM_LEASE_TTL_S=600
    ENV_SERVER_URL=http://...                   COLISEUM_BASE_URL=http://...

This shim re-exports the new names under their old names so existing
callers keep working; importing from this module emits a DeprecationWarning.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "openclaw_integration is deprecated; import from `coliseum` instead. "
    "See coliseum/README.md for the migration table.",
    DeprecationWarning,
    stacklevel=2,
)

from coliseum import (  # noqa: E402,F401
    ArenaClient as SreEnvClient,
    ArenaPool as LeasePool,
    app,
    create_arena_client as create_env_client,
)

__all__ = ["SreEnvClient", "LeasePool", "app", "create_env_client"]
