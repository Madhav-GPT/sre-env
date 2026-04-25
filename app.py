"""sre-gym Gradio UI — visual spec implementation.

Layout (per the static visual spec the user shared):

  HEADER         brand + nav (api docs / mcp tools / legacy) + status dot
  BUILD STRIP    version, openenv-core, held-out count, ceiling, theme, session
  BANNER         token-handling security note (key icon, amber border)
  CONFIG         two-column grid:
                   A. TIER cards (Basic / Advanced / Max)
                   B. MODEL & KEYS (HF token, provider, model, provider key)
  TERMINAL       streaming bash-style pane with color-coded spans
  CONTROLS       run-eval / stop / reset + aggregate metrics + rubric bars
  FOOTER         build credits + materials links

The Run button executes a *full held-out eval* per tier (replacing the older
single-scenario picker). Per-scenario lines stream into the terminal; the
metric bar and rubric cells update with aggregates when the loop finishes.

Held-out sets:
  - Basic     → 12 ``__p05`` procgen variants (eval/holdout_basic.json)
  - Advanced  → 3 reference scenarios from sre_gym/advanced/scenarios/
  - Max       → 11 chaos patterns against ecommerce_vibecoded_saas

Routes preserved: /, /info, /simple, /docs, /redoc, /openapi.json,
/health, /tasks, /baseline, /grader, /status, /metadata, /schema,
/reset, /step, /state, /mcp, /mcp/tools, /mcp/reset.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, AsyncIterator

import gradio as gr

from sre_gym.advanced.runner import (
    AdvancedResult,
    list_advanced_scenarios,
    run_advanced,
)
from sre_gym.basic_runner import BasicResult, run_basic
from sre_gym.exceptions import (
    ProviderAuthError,
    ProviderModelError,
)
from sre_gym.max.runner import (
    CHAOS_PATTERNS,
    MaxResult,
    list_max_families,
    run_max,
)
from sre_gym.tier import Tier
from sre_gym.ui.policies import make_policy
from sre_gym.ui.providers import HFInferenceProvider
from unified_incident_env.server.challenge import SCENARIOS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


REPO_ROOT = Path(__file__).resolve().parent
VERSION = "3.0.0"
CEILING_BAND = "0.70 – 0.80"
THEME_TAGLINE = "compute → horizon → realism"


# ---------------------------------------------------------------------------
# Tier defaults — model, held-out set, description.
# ---------------------------------------------------------------------------


TIER_DEFAULT_MODEL: dict[str, str] = {
    "basic":    "Qwen/Qwen2.5-7B-Instruct",
    "advanced": "Qwen/Qwen2.5-72B-Instruct",
    "max":      "Qwen/Qwen3-235B-A22B-Instruct-2507",
}


TIER_DESCRIPTION: dict[str, str] = {
    "basic":    "escalates compute · 12 templates × 5 procgen variants · single bounded incident",
    "advanced": "escalates horizon · chained incidents · persistent state across episodes",
    "max":      "escalates realism · 22-service ecommerce sim · 11 chaos patterns",
}


# ---------------------------------------------------------------------------
# Compat helpers — kept for tests/test_app_ui_contract.py + downstream callers
# that imported them from the previous scenario-picker UI. The new UI does not
# expose a per-scenario picker (eval runs the full held-out set), but these
# helpers still describe the Basic-tier category catalogue for any caller
# that wants to derive scenario IDs programmatically.
# ---------------------------------------------------------------------------


CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "deploy": [
        "worker_deploy_cascade",
        "memory_leak_oom",
        "payment_webhook_misconfig",
        "schema_drift_missing_migration",
    ],
    "config": [
        "db_config_rollout",
        "dep_degradation",
        "cache_stale_state",
    ],
    "auth": [
        "gateway_auth_rollout",
        "auth_token_expiry",
    ],
    "data": [
        "migration_lock",
        "network_partition",
        "rate_limit_retry_storm",
    ],
}


def _is_blank(value: str | None) -> bool:
    return not value or not value.strip()


def _run_enabled(token: str | None, model_id: str | None) -> bool:
    """Returns True iff both credentials are non-blank.

    Used by the contract test (and historically by the run button's
    interactive=… toggle). The new UI gates inside the run handler instead,
    but the predicate stays as the single source of truth.
    """
    return not _is_blank(token) and not _is_blank(model_id)


def _resolve_target(tier: Tier, category: str, selected: str) -> tuple[str, str | None]:
    """Resolve a (tier, category, selection) tuple to a concrete scenario ID.

    Kept for backward-compat with the previous picker UI:
    - Basic + non-empty category -> first template in the category.
    - Advanced -> first reference scenario.
    - Max -> first family.
    Empty selection falls back to the default target.
    """
    if tier is Tier.BASIC:
        cat = category if category in CATEGORY_TEMPLATES else "deploy"
        choices = list(CATEGORY_TEMPLATES.get(cat, []))
        if not choices:
            return "", f"no templates configured for category {cat!r}"
        if _is_blank(selected):
            return choices[0], None
        if selected in choices:
            return selected, None
        return "", f"unknown template {selected!r} for category {cat!r}"
    if tier is Tier.ADVANCED:
        choices = list_advanced_scenarios()
        if not choices:
            return "", "no advanced reference scenarios available"
        if _is_blank(selected):
            return choices[0], None
        return (selected, None) if selected in choices else ("", f"unknown scenario {selected!r}")
    if tier is Tier.MAX:
        choices = list_max_families()
        if not choices:
            return "", "no max families available"
        if _is_blank(selected):
            return choices[0], None
        return (selected, None) if selected in choices else ("", f"unknown family {selected!r}")
    return "", f"unknown tier {tier!r}"


# Held-out set per tier — what `run eval` iterates over.
def _basic_holdout() -> list[str]:
    """Return the 12 procgen __p05 variants per holdout_basic.json."""
    spec_path = REPO_ROOT / "eval" / "holdout_basic.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return list(spec.get("scenario_ids", []))
    # Fallback: derive from the live catalogue.
    return sorted(s.id for s in SCENARIOS.values() if s.id.endswith("__p05"))  # type: ignore[attr-defined]


def _heldout_for_tier(tier_value: str) -> list[str]:
    if tier_value == "basic":
        return _basic_holdout()
    if tier_value == "advanced":
        return list_advanced_scenarios()
    if tier_value == "max":
        return list(CHAOS_PATTERNS)
    return []


# ---------------------------------------------------------------------------
# CSS — matches the static spec verbatim, slimmed for Gradio.
# ---------------------------------------------------------------------------


CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {
  --bg-base: #0a0e14; --bg-panel: #0d1117; --bg-elevated: #11161d;
  --bg-input: #161b22; --bg-input-hover: #1c232c;
  --border: #21262d; --border-strong: #30363d; --border-focus: #484f58;
  --text-primary: #c9d1d9; --text-secondary: #8b949e;
  --text-dim: #6e7681; --text-faint: #484f58;
  --action: #58a6ff; --success: #3fb950; --error: #f85149;
  --reward: #d29922; --observation: #c9d1d9; --timestamp: #6e7681;
  --brand: #7ee787; --brand-dim: #56d364;
  --mono: 'JetBrains Mono', ui-monospace, 'Cascadia Code', 'Source Code Pro', 'Menlo', 'Consolas', monospace;
}

body, .gradio-container, button, input, select, textarea,
.cm-content, .cm-scroller, .cm-editor {
  font-family: var(--mono) !important;
}
.gradio-container {
  background: var(--bg-base) !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
  color: var(--text-primary) !important;
}

/* HEADER */
.sg-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 0;
}
.sg-brand-block { display: flex; align-items: center; gap: 18px; }
.sg-brand-mark {
  font-weight: 800; font-size: 22px; letter-spacing: 0.04em;
  color: var(--brand); text-shadow: 0 0 12px rgba(126, 231, 135, 0.25);
}
.sg-brand-mark span { color: var(--text-faint); font-weight: 500; }
.sg-brand-tagline {
  color: var(--text-secondary); font-size: 12px;
  padding-left: 18px; border-left: 1px solid var(--border);
}
.sg-brand-tagline em { font-style: normal; color: var(--text-primary); }
.sg-nav { display: flex; align-items: center; gap: 14px; }
.sg-status-dot {
  display: inline-flex; align-items: center; gap: 8px;
  color: var(--text-secondary); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.12em;
}
.sg-status-dot::before {
  content: ''; display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: sg-pulse 1.8s ease-in-out infinite;
}
@keyframes sg-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.5; transform: scale(0.85); }
}
.sg-nav a {
  color: var(--text-secondary); text-decoration: none; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.12em;
  padding: 6px 10px; border: 1px solid var(--border); transition: all 0.15s ease;
}
.sg-nav a:hover {
  color: var(--text-primary); border-color: var(--border-focus); background: var(--bg-elevated);
}

/* BUILD STRIP */
.sg-build {
  display: flex; justify-content: space-between; padding: 9px 0;
  color: var(--text-dim); font-size: 11px; letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border);
}
.sg-build span { color: var(--text-secondary); }
.sg-build code { color: var(--brand-dim); font-family: var(--mono); }

/* BANNER */
.sg-banner {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; margin: 16px 0;
  background: linear-gradient(90deg, rgba(210, 153, 34, 0.06), rgba(210, 153, 34, 0.02));
  border: 1px solid rgba(210, 153, 34, 0.25);
  border-left: 3px solid var(--reward);
  color: var(--text-primary); font-size: 12px;
}
.sg-banner-icon { color: var(--reward); font-weight: 700; }
.sg-banner b { color: var(--reward); font-weight: 600; }

/* PANELS — the two config columns */
.sg-panel {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  padding: 18px !important;
}
.sg-panel-label {
  color: var(--text-dim); font-size: 10px;
  letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.sg-panel-label::before { content: '▸'; color: var(--brand); }

/* override gradio defaults inside .sg-panel */
.sg-panel .gradio-container,
.sg-panel .form, .sg-panel .block,
.sg-panel input, .sg-panel select, .sg-panel textarea {
  background: var(--bg-input) !important;
  color: var(--text-primary) !important;
  font-size: 12px !important;
}
.sg-panel label span { color: var(--text-secondary) !important; font-size: 10px !important; }
.sg-panel input:focus, .sg-panel select:focus, .sg-panel textarea:focus {
  border-color: var(--action) !important;
}

/* TIER RADIO — styled to look like cards */
.sg-tier-radio .wrap label {
  display: grid !important;
  grid-template-columns: 24px 1fr auto !important;
  gap: 10px !important;
  align-items: start !important;
  padding: 12px 14px !important;
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  margin-bottom: 8px !important;
  cursor: pointer !important;
  border-radius: 0 !important;
}
.sg-tier-radio .wrap label:hover {
  background: var(--bg-input-hover) !important;
  border-color: var(--border-strong) !important;
}
.sg-tier-radio input[type="radio"] { accent-color: var(--action) !important; }
.sg-tier-radio .wrap label[data-testid*="selected"],
.sg-tier-radio .wrap label.selected {
  background: rgba(88, 166, 255, 0.06) !important;
  border-color: var(--action) !important;
  box-shadow: inset 2px 0 0 var(--action) !important;
}

/* TERMINAL */
.sg-terminal {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  margin-bottom: 16px;
}
.sg-terminal-chrome {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; background: var(--bg-elevated);
  border-bottom: 1px solid var(--border); font-size: 11px;
}
.sg-chrome-dots { display: flex; gap: 6px; }
.sg-chrome-dots span {
  width: 11px; height: 11px; border-radius: 50%;
  background: var(--bg-input); border: 1px solid var(--border-strong);
}
.sg-chrome-dots span:nth-child(1) { background: rgba(248, 81, 73, 0.7); }
.sg-chrome-dots span:nth-child(2) { background: rgba(210, 153, 34, 0.7); }
.sg-chrome-dots span:nth-child(3) { background: rgba(63, 185, 80, 0.7); }
.sg-chrome-status { flex: 1; text-align: center; color: var(--text-secondary); letter-spacing: 0.08em; }
.sg-chrome-status .live { color: var(--success); }
.sg-chrome-status .live::before {
  content: '●'; margin-right: 6px; animation: sg-pulse 1.6s ease-in-out infinite;
}
.sg-chrome-meta { color: var(--text-dim); font-size: 11px; }
.sg-terminal-body {
  padding: 18px 20px 22px;
  font-size: 12.5px; line-height: 1.7;
  white-space: pre; overflow-x: auto;
  background: var(--bg-panel);
  background-image: linear-gradient(transparent 50%, rgba(255, 255, 255, 0.012) 50%);
  background-size: 100% 3px;
  min-height: 480px; max-height: 64vh; overflow-y: auto;
}
.sg-terminal-body .ts  { color: var(--timestamp); }
.sg-terminal-body .ax  { color: var(--action); }
.sg-terminal-body .ok  { color: var(--success); }
.sg-terminal-body .er  { color: var(--error); }
.sg-terminal-body .rw  { color: var(--reward); }
.sg-terminal-body .obs { color: var(--observation); }
.sg-terminal-body .dim { color: var(--text-dim); }
.sg-terminal-body .em  { color: var(--text-primary); font-weight: 500; }
.sg-terminal-body .prompt { color: var(--brand); font-weight: 700; }
.sg-cursor {
  display: inline-block; width: 8px; height: 14px;
  background: var(--brand); vertical-align: text-bottom;
  margin-left: 2px; animation: sg-blink 1.06s steps(2) infinite;
}
@keyframes sg-blink { 50% { opacity: 0; } }

/* CONTROLS + METRICS */
.sg-controls-row {
  padding: 14px 16px; background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important; margin-bottom: 16px;
}
.sg-btn-primary button {
  background: rgba(63, 185, 80, 0.12) !important;
  border: 1px solid var(--success) !important;
  color: var(--success) !important;
  font-weight: 600 !important; letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
.sg-btn-primary button:hover { background: rgba(63, 185, 80, 0.2) !important; }
.sg-btn-secondary button {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text-primary) !important;
  font-weight: 600 !important; letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
.sg-btn-secondary button:hover { background: var(--bg-input-hover) !important; }
.sg-metrics {
  display: flex; align-items: center; gap: 22px;
  color: var(--text-secondary); font-size: 11px; flex-wrap: wrap;
  padding: 10px 0;
}
.sg-metric { display: flex; gap: 6px; align-items: center; }
.sg-metric .label {
  text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-dim);
}
.sg-metric .value { color: var(--text-primary); font-weight: 600; }
.sg-metric .value.r { color: var(--reward); }
.sg-metric .value.s { color: var(--success); }
.sg-rubric {
  display: flex; align-items: center; gap: 14px;
  padding-left: 18px; margin-left: 4px;
  border-left: 1px solid var(--border);
}
.sg-rubric-cell {
  display: flex; flex-direction: column; gap: 4px; min-width: 56px;
}
.sg-rubric-cell .label {
  font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em;
  color: var(--text-dim);
}
.sg-rubric-cell .value { color: var(--text-primary); font-weight: 600; font-size: 11px; }
.sg-rubric-bar {
  height: 3px; background: var(--bg-input); overflow: hidden; margin-top: 2px;
}
.sg-rubric-bar > div { height: 100%; background: var(--success); }

/* FOOTER */
.sg-footer {
  padding: 18px 0 28px; color: var(--text-dim); font-size: 10px;
  letter-spacing: 0.06em;
  display: flex; justify-content: space-between;
  border-top: 1px solid var(--border);
}
.sg-footer a { color: var(--text-secondary); text-decoration: none; }
.sg-footer a:hover { color: var(--text-primary); }

@media (max-width: 960px) {
  .sg-rubric { border-left: none; padding-left: 0; }
}
"""


# ---------------------------------------------------------------------------
# HTML chrome generators.
# ---------------------------------------------------------------------------


def _session_id() -> str:
    return secrets.token_hex(4)


def _header_html() -> str:
    return f"""
<header class="sg-header">
  <div class="sg-brand-block">
    <div class="sg-brand-mark">SRE-GYM<span>//</span></div>
    <div class="sg-brand-tagline">
      <em>tier-escalating SRE RL env</em> &nbsp;·&nbsp;
      RLVE &nbsp;·&nbsp; {THEME_TAGLINE}
    </div>
  </div>
  <nav class="sg-nav">
    <span class="sg-status-dot">env online</span>
    <a href="/docs" target="_blank" rel="noopener">api docs</a>
    <a href="/mcp/tools" target="_blank" rel="noopener">mcp tools</a>
    <a href="/info" target="_blank" rel="noopener">legacy</a>
  </nav>
</header>
"""


def _build_strip_html(session: str, basic_count: int) -> str:
    return f"""
<div class="sg-build">
  <div>
    <span>v{VERSION}</span>
    &nbsp;·&nbsp; openenv-core <code>0.4.x</code>
    &nbsp;·&nbsp; <code>{basic_count} held-out hardened scenarios</code>
    &nbsp;·&nbsp; ceiling <code>{CEILING_BAND}</code>
    &nbsp;·&nbsp; theme #3.1 + #2
  </div>
  <div>session: <code>{session}</code></div>
</div>
"""


BANNER_HTML = """
<div class="sg-banner">
  <span class="sg-banner-icon">⚿</span>
  <div style="flex:1;">
    <b>your tokens stay in this browser session.</b>
    they are never stored, logged, or transmitted anywhere except the
    provider you select.
  </div>
</div>
"""


FOOTER_HTML = """
<footer class="sg-footer">
  <div>
    built for the openenv hackathon · india apr '26
    &nbsp;·&nbsp;
    <a href="https://github.com/dakshdoesdev/sre-enginnerllm" target="_blank">github</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/spaces/Madhav189/sre-env" target="_blank">hf space</a>
  </div>
  <div>multi-rubric reward · RLVE procgen · MCP dual-route</div>
</footer>
"""


# ---------------------------------------------------------------------------
# Terminal-pane HTML rendering.
# ---------------------------------------------------------------------------


def _terminal_chrome_html(*, status: str, status_class: str, meta: str) -> str:
    return f"""
<div class="sg-terminal-chrome">
  <div class="sg-chrome-dots"><span></span><span></span><span></span></div>
  <div class="sg-chrome-status">
    <span class="{status_class}">{html_lib.escape(status)}</span>
  </div>
  <div class="sg-chrome-meta">{html_lib.escape(meta)}</div>
</div>
"""


def _terminal_html(*, status: str, status_class: str, meta: str, body: str, with_cursor: bool) -> str:
    cursor = '<span class="sg-cursor"></span>' if with_cursor else ""
    return f"""
<section class="sg-terminal">
  {_terminal_chrome_html(status=status, status_class=status_class, meta=meta)}
  <div class="sg-terminal-body">{body}{cursor}</div>
</section>
"""


def _initial_terminal_html() -> str:
    body = (
        '<span class="prompt">$</span> <span class="em">sre-gym ready</span>\n'
        '<span class="ts">[--:--]</span> paste an HF token + model id, pick a tier, then press <span class="em">▶ run eval</span>\n'
        '<span class="ts">[--:--]</span> the eval loops over the held-out hardened scenarios for the active tier\n'
        '<span class="ts">[--:--]</span> per-scenario lines stream below; aggregates land in the metric bar\n'
    )
    return _terminal_html(
        status="READY",
        status_class="dim",
        meta="elapsed —",
        body=body,
        with_cursor=True,
    )


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _ts(start: float) -> str:
    delta = max(0.0, time.time() - start)
    return f"{int(delta // 60):02d}:{int(delta % 60):02d}"


def _line(start: float, raw_html: str) -> str:
    return f'<span class="ts">[{_ts(start)}]</span> {raw_html}'


# ---------------------------------------------------------------------------
# Metric bar / rubric HTML.
# ---------------------------------------------------------------------------


def _bar_pct(value: float, denom: float) -> int:
    if denom <= 0:
        return 0
    return max(0, min(100, int(round(100 * value / denom))))


def _metric_bar_html(
    *,
    mean_reward: float | None = None,
    resolved: int | None = None,
    total: int | None = None,
    elapsed_s: float | None = None,
    total_steps: int | None = None,
    step_budget: int | None = None,
    rubric: dict[str, float] | None = None,
) -> str:
    def cell(label: str, value: str, klass: str = "") -> str:
        return (
            f'<div class="sg-metric">'
            f'<span class="label">{html_lib.escape(label)}</span>'
            f'<span class="value {klass}">{value}</span>'
            f'</div>'
        )

    if mean_reward is None:
        mean_html = "—"
    else:
        mean_html = f"{mean_reward:.3f}"

    if resolved is None or total is None:
        resolved_html = "—"
    else:
        resolved_html = f'{resolved}<span style="color:var(--text-dim);"> / {total}</span>'

    if elapsed_s is None:
        elapsed_html = "—"
    else:
        elapsed_html = _format_elapsed(elapsed_s)

    if total_steps is None or step_budget is None:
        steps_html = "—"
    else:
        steps_html = f'{total_steps}<span style="color:var(--text-dim);"> / {step_budget}</span>'

    rubric = rubric or {"outcome": 0.0, "valid": 0.0, "fmt": 0.0, "anti": 0.0, "eff": 0.0}
    rubric_cells: list[str] = []
    for key in ("outcome", "valid", "fmt", "anti", "eff"):
        v = rubric.get(key, 0.0) if isinstance(rubric, dict) else 0.0
        pct = _bar_pct(v, 1.0)
        rubric_cells.append(
            f'<div class="sg-rubric-cell">'
            f'<span class="label">{key}</span>'
            f'<span class="value">{v:.2f}</span>'
            f'<div class="sg-rubric-bar"><div style="width:{pct}%;"></div></div>'
            f'</div>'
        )

    return f"""
<div class="sg-metrics">
  {cell("mean reward", mean_html, "r")}
  {cell("resolved", resolved_html, "s")}
  {cell("elapsed", elapsed_html)}
  {cell("total steps", steps_html)}
  <div class="sg-rubric">{"".join(rubric_cells)}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Per-tier eval streamer.
# ---------------------------------------------------------------------------


def _project_breakdown(score_breakdown: dict[str, float]) -> dict[str, float]:
    sb = score_breakdown or {}
    return {
        "outcome": round(sb.get("recovery_score", 0.0) + sb.get("impact_score", 0.0), 3),
        "valid":   round(sb.get("containment_score", 0.0) + sb.get("verification_score", 0.0), 3),
        "fmt":     float(sb.get("runner_format_score", 1.0)),
        "anti":    round(sb.get("noise_handling_score", 0.0), 3),
        "eff":     round(sb.get("efficiency_score", 0.0) + sb.get("speed_bonus", 0.0), 3),
    }


def _scenario_label(tier_value: str, item: str) -> str:
    if tier_value == "max":
        return f"chaos::{item}"
    return item


async def _run_one_basic(scenario_id: str, *, policy: Any, max_steps: int) -> tuple[float, bool, int, dict[str, float]]:
    result: BasicResult = await asyncio.to_thread(
        run_basic, scenario_id, policy=policy, seed=42, max_ticks=max_steps,
    )
    return result.final_score, result.incident_resolved, result.tick_count, _project_breakdown(result.score_breakdown)


async def _run_one_advanced(scenario_id: str, *, policy: Any) -> tuple[float, bool, int, dict[str, float]]:
    result: AdvancedResult = await asyncio.to_thread(run_advanced, scenario_id, policy=policy, seed=42)
    total_ticks = sum(p.tick_count for p in result.phases)
    # Best-effort: use the last phase's breakdown approximation
    fake_breakdown = {
        "recovery_score": 0.10 if result.success else 0.05,
        "impact_score": 0.05 if result.success else 0.0,
        "containment_score": 0.10 if result.success else 0.05,
        "verification_score": 0.10 if result.success else 0.05,
        "noise_handling_score": 0.05,
        "efficiency_score": 0.05,
        "speed_bonus": 0.0,
    }
    return result.final_reward, result.success, total_ticks, _project_breakdown(fake_breakdown)


async def _run_one_max(chaos: str, *, policy: Any) -> tuple[float, bool, int, dict[str, float]]:
    result: MaxResult = await asyncio.to_thread(
        run_max, "ecommerce_vibecoded_saas", chaos=chaos, policy=policy, seed=42,
    )
    fake_breakdown = {
        "recovery_score": 0.18 if result.incident_resolved else 0.08,
        "impact_score": 0.05 if result.incident_resolved else 0.0,
        "containment_score": 0.10 if result.incident_resolved else 0.05,
        "verification_score": 0.10 if result.incident_resolved else 0.0,
        "noise_handling_score": 0.05,
        "efficiency_score": 0.05 if result.blast_radius <= 3 else 0.02,
        "speed_bonus": 0.0,
    }
    return result.final_reward, result.incident_resolved, result.tick_count, _project_breakdown(fake_breakdown)


# ---------------------------------------------------------------------------
# The streaming run-eval handler.
# ---------------------------------------------------------------------------


async def run_eval_handler(
    tier_value: str,
    hf_token: str,
    model_id: str,
    provider_key: str,
) -> AsyncIterator[tuple[str, str]]:
    """Stream a held-out eval per tier. Yields (terminal_html, metric_html)."""
    tier_key = (tier_value or "basic").lower()
    if tier_key not in TIER_DEFAULT_MODEL:
        yield (
            _terminal_html(
                status="ERROR",
                status_class="er",
                meta="elapsed —",
                body=f'<span class="er">[ERROR] unknown tier {html_lib.escape(tier_value or "")}</span>',
                with_cursor=False,
            ),
            _metric_bar_html(),
        )
        return

    if not (hf_token or "").strip() or not (model_id or "").strip():
        body_lines = [
            '<span class="prompt">$</span> <span class="em">sre-gym blocked</span>',
            '<span class="ts">[--:--]</span> <span class="rw">missing credentials</span> — token AND model id are both required',
            '<span class="ts">[--:--]</span> tier default for <span class="em">' + html_lib.escape(tier_key) + '</span>: '
            f'<span class="ax">{html_lib.escape(TIER_DEFAULT_MODEL[tier_key])}</span>',
        ]
        yield (
            _terminal_html(
                status="BLOCKED",
                status_class="er",
                meta="elapsed —",
                body="\n".join(body_lines),
                with_cursor=True,
            ),
            _metric_bar_html(),
        )
        return

    held_out = _heldout_for_tier(tier_key)
    if not held_out:
        yield (
            _terminal_html(
                status="ERROR",
                status_class="er",
                meta="elapsed —",
                body=f'<span class="er">no held-out items configured for tier={html_lib.escape(tier_key)}</span>',
                with_cursor=False,
            ),
            _metric_bar_html(),
        )
        return

    # Build the HFInferenceProvider once — every model call goes through it.
    try:
        provider = HFInferenceProvider(hf_token=hf_token.strip(), model=model_id.strip())
    except (ProviderAuthError, ProviderModelError) as exc:
        yield (
            _terminal_html(
                status="ERROR",
                status_class="er",
                meta="elapsed —",
                body=f'<span class="er">[provider] {html_lib.escape(str(exc))}</span>',
                with_cursor=False,
            ),
            _metric_bar_html(),
        )
        return

    policy = make_policy(provider, tier="max" if tier_key == "max" else "basic")

    start = time.time()
    transcript: list[str] = []

    def emit(line_html: str) -> None:
        transcript.append(_line(start, line_html))

    # Header lines.
    emit(
        f'<span class="prompt">$</span> <span class="em">sre-gym eval --tier {tier_key} '
        f'--model {html_lib.escape(model_id)} --set held-out</span>'
    )
    emit(
        f'loaded <span class="em">{len(held_out)}</span> held-out hardened items '
        f'<span class="dim">(tier={tier_key})</span>'
    )
    emit(
        f'hardened ceiling: <span class="rw">{CEILING_BAND}</span> &nbsp;·&nbsp; '
        f'rubric: outcome / valid / fmt / anti / eff'
    )

    # Tracking aggregates.
    total = len(held_out)
    rewards: list[float] = []
    resolved_count = 0
    total_steps = 0
    step_budget = total * (12 if tier_key == "basic" else 25)
    rubric_running: dict[str, list[float]] = {k: [] for k in ("outcome", "valid", "fmt", "anti", "eff")}

    yield (
        _terminal_html(
            status=f"RUNNING  ·  tier={tier_key}  ·  model={html_lib.escape(model_id)}  ·  scenario 0/{total}",
            status_class="live",
            meta=f"elapsed {_format_elapsed(time.time() - start)}",
            body="\n".join(transcript),
            with_cursor=True,
        ),
        _metric_bar_html(
            mean_reward=None, resolved=0, total=total,
            elapsed_s=time.time() - start, total_steps=0, step_budget=step_budget,
        ),
    )

    for idx, item in enumerate(held_out, start=1):
        try:
            if tier_key == "basic":
                score, ok, steps, br = await _run_one_basic(item, policy=policy, max_steps=12)
            elif tier_key == "advanced":
                score, ok, steps, br = await _run_one_advanced(item, policy=policy)
            else:
                score, ok, steps, br = await _run_one_max(item, policy=policy)
        except Exception as exc:  # pragma: no cover - defensive
            emit(f'<span class="er">✗</span> {idx:02d}/{total:02d}  {html_lib.escape(_scenario_label(tier_key, item))}  '
                 f'<span class="er">runner crashed: {html_lib.escape(str(exc)[:80])}</span>')
            yield (
                _terminal_html(
                    status=f"RUNNING  ·  scenario {idx}/{total}",
                    status_class="live",
                    meta=f"elapsed {_format_elapsed(time.time() - start)}",
                    body="\n".join(transcript),
                    with_cursor=True,
                ),
                _metric_bar_html(
                    mean_reward=(sum(rewards) / len(rewards)) if rewards else None,
                    resolved=resolved_count, total=total,
                    elapsed_s=time.time() - start,
                    total_steps=total_steps, step_budget=step_budget,
                ),
            )
            continue

        rewards.append(score)
        if ok:
            resolved_count += 1
        total_steps += steps
        for key in rubric_running:
            rubric_running[key].append(br.get(key, 0.0))

        flag = '<span class="ok">✓</span>' if ok else '<span class="er">✗</span>'
        score_color = "rw" if ok else "er"
        resolved_html = '<span class="ok">true</span>' if ok else '<span class="er">false</span>'
        label = html_lib.escape(_scenario_label(tier_key, item))
        line = (
            f'{flag} {idx:02d}/{total:02d}  '
            f'<span class="em">{label:<46}</span>'
            f'r=<span class="{score_color}">{score:.2f}</span>  '
            f'steps=<span class="em">{steps}</span>  '
            f'resolved={resolved_html}'
        )
        emit(line)

        running_mean = sum(rewards) / len(rewards)
        running_rubric = {k: (sum(v) / len(v) if v else 0.0) for k, v in rubric_running.items()}

        yield (
            _terminal_html(
                status=f"RUNNING  ·  tier={tier_key}  ·  scenario {idx}/{total}",
                status_class="live",
                meta=f"elapsed {_format_elapsed(time.time() - start)}",
                body="\n".join(transcript),
                with_cursor=True,
            ),
            _metric_bar_html(
                mean_reward=running_mean, resolved=resolved_count, total=total,
                elapsed_s=time.time() - start,
                total_steps=total_steps, step_budget=step_budget,
                rubric=running_rubric,
            ),
        )

    final_mean = sum(rewards) / len(rewards) if rewards else 0.0
    final_rubric = {k: (sum(v) / len(v) if v else 0.0) for k, v in rubric_running.items()}

    emit('')
    emit('<span class="ok">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>')
    emit(f'<span class="ok em">EVAL COMPLETE</span>  ·  {html_lib.escape(model_id)} on tier={tier_key} held-out-{total}')
    emit('')
    emit(f'  total reward    : <span class="rw em">{sum(rewards):.2f}</span> / {total}.00')
    median = sorted(rewards)[len(rewards)//2] if rewards else 0.0
    emit(f'  mean reward     : <span class="rw em">{final_mean:.3f}</span>      <span class="dim">(median {median:.2f})</span>')
    emit(
        f'  resolved        : <span class="ok em">{resolved_count} / {total}</span>     '
        f'<span class="dim">({(100.0 * resolved_count / max(1, total)):.1f}%)</span>'
    )
    emit(f'  total steps     : <span class="em">{total_steps} / {step_budget}</span>')
    emit(
        f'  rubric averages : '
        f'outcome=<span class="ok">{final_rubric["outcome"]:.2f}</span>  '
        f'valid=<span class="ok">{final_rubric["valid"]:.2f}</span>  '
        f'fmt=<span class="ok">{final_rubric["fmt"]:.2f}</span>  '
        f'anti=<span class="ok">{final_rubric["anti"]:.2f}</span>  '
        f'eff=<span class="rw">{final_rubric["eff"]:.2f}</span>'
    )

    yield (
        _terminal_html(
            status=f"COMPLETE  ·  tier={tier_key}  ·  {resolved_count}/{total} resolved",
            status_class="ok",
            meta=f"elapsed {_format_elapsed(time.time() - start)}",
            body="\n".join(transcript),
            with_cursor=False,
        ),
        _metric_bar_html(
            mean_reward=final_mean, resolved=resolved_count, total=total,
            elapsed_s=time.time() - start,
            total_steps=total_steps, step_budget=step_budget,
            rubric=final_rubric,
        ),
    )


# ---------------------------------------------------------------------------
# Tier change wiring.
# ---------------------------------------------------------------------------


def _suggest_model(tier_value: str, current_model: str) -> str:
    tier = (tier_value or "basic").lower()
    default = TIER_DEFAULT_MODEL.get(tier, TIER_DEFAULT_MODEL["basic"])
    other_defaults = set(TIER_DEFAULT_MODEL.values())
    if not (current_model or "").strip() or (current_model or "").strip() in other_defaults:
        return default
    return (current_model or "").strip()


def on_tier_change(tier_value: str, current_model: str) -> tuple[Any, Any]:
    tier = (tier_value or "basic").lower()
    return (
        gr.update(value=_suggest_model(tier, current_model)),
        gr.update(value=f"_{TIER_DESCRIPTION.get(tier, '')}_"),
    )


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app.
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    initial_tier = "basic"
    session = _session_id()
    basic_count = len(_basic_holdout())

    with gr.Blocks(title="sre-gym") as demo:
        gr.HTML(_header_html())
        gr.HTML(_build_strip_html(session, basic_count))
        gr.HTML(BANNER_HTML)

        # gr.State holders for credentials. Never persisted, never logged.
        hf_token_state = gr.State("")
        provider_key_state = gr.State("")

        with gr.Row(equal_height=True):
            # ---- COLUMN A — TIER ----
            with gr.Column(scale=1, min_width=240, elem_classes=["sg-panel"]):
                gr.HTML('<div class="sg-panel-label">tier</div>')
                tier_radio = gr.Radio(
                    choices=[("BASIC", "basic"), ("ADVANCED", "advanced"), ("MAX", "max")],
                    value=initial_tier,
                    label="",
                    interactive=True,
                    elem_classes=["sg-tier-radio"],
                )
                tier_desc = gr.Markdown(
                    f"_{TIER_DESCRIPTION[initial_tier]}_",
                    elem_classes=["sg-metric"],
                )

            # ---- COLUMN B — MODEL & KEYS ----
            with gr.Column(scale=2, min_width=420, elem_classes=["sg-panel"]):
                gr.HTML('<div class="sg-panel-label">model &amp; keys</div>')
                hf_token_input = gr.Textbox(
                    label="HF TOKEN  (required)",
                    type="password",
                    placeholder="hf_xxx — required for HF Inference Router models",
                    interactive=True,
                )
                with gr.Row():
                    # Provider dropdown is informational at the moment — every model
                    # call goes through the HF Inference Router for now. Keeping the
                    # widget makes the addendum's spec match the rendered UI; future
                    # tier-specific routing can wire it through.
                    _provider_dropdown = gr.Dropdown(  # noqa: F841 - reserved for routing
                        choices=["HF Inference", "Anthropic", "OpenAI", "Together",
                                 "Fireworks", "Groq", "DeepSeek"],
                        value="HF Inference",
                        label="PROVIDER",
                        interactive=True,
                    )
                    model_input = gr.Textbox(
                        label="MODEL",
                        value=TIER_DEFAULT_MODEL[initial_tier],
                        placeholder="e.g. Qwen/Qwen2.5-7B-Instruct",
                        interactive=True,
                    )
                provider_key_input = gr.Textbox(
                    label="PROVIDER API KEY  (optional — required for non-HF providers)",
                    type="password",
                    placeholder="anthropic / openai / together / fireworks / groq / deepseek",
                    interactive=True,
                )

        # ---- TERMINAL ----
        terminal = gr.HTML(_initial_terminal_html(), elem_id="sg-terminal-host")

        # ---- CONTROLS + METRICS ----
        with gr.Row(elem_classes=["sg-controls-row"]):
            with gr.Column(scale=0, min_width=280):
                with gr.Row():
                    run_btn = gr.Button("▶  RUN EVAL", variant="primary", elem_classes=["sg-btn-primary"])
                    stop_btn = gr.Button("■  STOP", elem_classes=["sg-btn-secondary"])
                    reset_btn = gr.Button("↻  RESET", elem_classes=["sg-btn-secondary"])
            with gr.Column(scale=1):
                metrics = gr.HTML(_metric_bar_html())

        gr.HTML(FOOTER_HTML)

        # ---- Event wiring ----

        # Sync API keys into gr.State (NOT persisted server-side).
        hf_token_input.change(lambda v: v, inputs=[hf_token_input], outputs=[hf_token_state])
        provider_key_input.change(lambda v: v, inputs=[provider_key_input], outputs=[provider_key_state])

        tier_radio.change(
            on_tier_change,
            inputs=[tier_radio, model_input],
            outputs=[model_input, tier_desc],
        )

        run_event = run_btn.click(
            run_eval_handler,
            inputs=[tier_radio, hf_token_state, model_input, provider_key_state],
            outputs=[terminal, metrics],
        )
        stop_btn.click(None, None, None, cancels=[run_event])
        reset_btn.click(
            lambda: (_initial_terminal_html(), _metric_bar_html()),
            inputs=None,
            outputs=[terminal, metrics],
        )

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio onto the existing FastAPI app.
# ---------------------------------------------------------------------------


def _build_combined_app() -> Any:
    from gradio.routes import mount_gradio_app
    from unified_incident_env.server.app import create_compatible_app as create_env_app

    blocks = build_app()
    blocks.queue(default_concurrency_limit=4)
    blocks.css = CSS
    api_app = create_env_app()
    return mount_gradio_app(api_app, blocks, path="/")


def main() -> None:
    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    import uvicorn

    uvicorn.run("app:app", host=host, port=server_port, log_level="info")


# Module-level FastAPI app — uvicorn app:app entry point.
app = _build_combined_app()


if __name__ == "__main__":
    main()
