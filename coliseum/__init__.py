"""Coliseum — parallel-rollout pool server for sre-gym Triage.

Exposes the Triage tier through a lease-based HTTP contract
(``allocate / heartbeat / reset / exec_tool / evaluate / close``) so a
GRPO trainer's rollout side can drive the env without holding an in-process
``UnifiedIncidentEnvironment`` per worker.

Public surface:

    from coliseum import ArenaClient, create_arena_client, ArenaPool

The contract shape is the lease-pool pattern that's standard in
parallel-rollout RL frameworks; nothing here is bound to a specific trainer
implementation.
"""

from __future__ import annotations

from .client import ArenaClient, create_arena_client
from .server import ArenaPool, app

__all__ = ["ArenaClient", "ArenaPool", "app", "create_arena_client"]
