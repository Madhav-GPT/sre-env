"""Asserts skill/verified-runbooks/ has a runbook for every Basic template.

Catches the case where a contributor adds a new template but forgets to seed
its runbook. Drafts (status: draft) count — they're known incomplete but
provide enough scaffolding for the Claude Code skill to pre-load decision
trees on subsequent solves.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNBOOK_DIR = REPO_ROOT / "skill" / "verified-runbooks"

REQUIRED_TEMPLATES: list[str] = [
    # v2 templates (verified runbooks)
    "worker_deploy_cascade",
    "db_config_rollout",
    "gateway_auth_rollout",
    # Vibe-coded SaaS extension band (draft runbooks)
    "payment_webhook_misconfig",
    "schema_drift_missing_migration",
    "cache_stale_state",
    # Round-2 Basic-tier additions (draft runbooks)
    "dep_degradation",
    "memory_leak_oom",
    "auth_token_expiry",
    "network_partition",
    "rate_limit_retry_storm",
    "migration_lock",
]


def test_runbook_directory_exists() -> None:
    assert RUNBOOK_DIR.is_dir(), f"runbook directory missing: {RUNBOOK_DIR}"


@pytest.mark.parametrize("template_id", REQUIRED_TEMPLATES)
def test_runbook_present_for_template(template_id: str) -> None:
    """Every Basic-tier template must have a corresponding .md runbook."""
    rb = RUNBOOK_DIR / f"{template_id}.md"
    assert rb.is_file(), (
        f"missing runbook for {template_id}. Stub one at {rb} with the "
        f"5-section structure (Symptoms / Decision tree / Action sequence / "
        f"Success criteria / Rollback notes). See docs/SCENARIO_AUTHORING.md."
    )


@pytest.mark.parametrize("template_id", REQUIRED_TEMPLATES)
def test_runbook_has_required_sections(template_id: str) -> None:
    """Every runbook must have the 5 canonical sections."""
    rb = RUNBOOK_DIR / f"{template_id}.md"
    text = rb.read_text(encoding="utf-8")
    required_sections = [
        "## Symptoms",
        "## Decision tree",
        "## Action sequence",
        "## Success criteria",
        "## Rollback",
    ]
    for section in required_sections:
        assert section in text, f"{template_id}.md missing section: {section}"


@pytest.mark.parametrize("template_id", REQUIRED_TEMPLATES)
def test_runbook_has_frontmatter(template_id: str) -> None:
    """Every runbook must declare template_id + status in YAML frontmatter."""
    rb = RUNBOOK_DIR / f"{template_id}.md"
    text = rb.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{template_id}.md must start with YAML frontmatter"
    fm_end = text.find("\n---\n", 4)
    assert fm_end > 0, f"{template_id}.md frontmatter not closed"
    fm = text[4:fm_end]
    assert f"template_id: {template_id}" in fm, f"{template_id}.md frontmatter must declare template_id"
    assert "status:" in fm, f"{template_id}.md frontmatter must declare status (draft|verified)"


def test_runbook_count_matches_templates() -> None:
    """The runbook directory must contain exactly the templates we require, no orphans."""
    on_disk = {p.stem for p in RUNBOOK_DIR.glob("*.md")}
    required = set(REQUIRED_TEMPLATES)
    extra = on_disk - required
    missing = required - on_disk
    assert not missing, f"runbooks missing for templates: {sorted(missing)}"
    assert not extra, (
        f"orphan runbooks for unknown templates: {sorted(extra)}. "
        f"Either add the template or remove the stale runbook."
    )
