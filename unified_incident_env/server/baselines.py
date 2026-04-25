"""Tiny helper for declarative baseline-step authoring.

The original 6 templates inline ``BaselineStep(action=UnifiedIncidentAction(...), rationale=...)``
in their builder functions. The 6 round-2 templates use this ``_ba()`` shortcut so each
baseline step fits on one line.

This module is intentionally trivial. Anything beyond ``_ba`` belongs in challenge.py.
"""

from __future__ import annotations

from typing import Any

from ..models import BaselineStep, UnifiedIncidentAction


def _ba(action_type: str, *, rationale: str = "", **fields: Any) -> BaselineStep:
    """Build a ``BaselineStep`` with a single positional action_type and inline rationale.

    Examples
    --------
    >>> _ba("query_logs", service="worker", rationale="...")
    >>> _ba("submit_hypothesis", hypothesis={...}, rationale="...")
    >>> _ba("declare_resolved", rationale="...")
    """
    return BaselineStep(
        action=UnifiedIncidentAction(action_type=action_type, **fields),
        rationale=rationale,
    )
