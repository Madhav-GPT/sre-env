"""Terminal-style Gradio UI for sre-gym (UI Build Addendum implementation).

The HF Space serves:
  /                 Gradio terminal UI (this file)
  /info /simple     legacy markdown landing page
  /docs /redoc      Swagger / ReDoc
  /health /tasks /baseline /grader /status /metadata /schema
                    OpenEnv + extension routes
  /reset /step /state              OpenEnv contract
  /mcp /mcp/tools /mcp/reset       JSON-RPC 2.0 MCP dual-route
  /openapi.json     OpenAPI spec

All on port 7860. Gradio is mounted onto the existing FastAPI server via
``gradio.routes.mount_gradio_app(api_app, blocks, path="/")``; no FastAPI
route is modified.

Tokens (HF / Anthropic / OpenAI / Groq / Together / Fireworks / DeepSeek)
live ONLY in ``gr.State`` for the duration of the user's browser session.
They are never persisted, logged, or echoed back into the terminal pane.

Run locally::

    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import gradio as gr

from sre_gym.exceptions import (
    ProviderAuthError,
    SREGymError,
)
from sre_gym.max.runner import CHAOS_PATTERNS, list_max_families
from sre_gym.advanced.runner import list_advanced_scenarios
from sre_gym.tier import Tier
from sre_gym.ui.providers import DummyProvider, Provider
from sre_gym.ui.router import (
    ModelEntry,
    ProviderKind,
    build_provider,
    find_entry,
    models_for_tier,
)
from sre_gym.ui.runner import stream_episode
from unified_incident_env.server.challenge import list_scenarios

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Basic-tier category groupings — used by the "category" dropdown per
# UI Build Addendum §3.1 (Column C). Each Basic template is assigned to one
# of {deploy, config, auth, data} based on its dominant failure family.
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


def _category_for_template(template_id: str) -> str | None:
    base = template_id.split("__")[0]
    for cat, ids in CATEGORY_TEMPLATES.items():
        if base in ids:
            return cat
    return None


def _templates_in_category(category: str) -> list[str]:
    """Return all scenario IDs (template + procgen variants) within a category."""
    base_ids = CATEGORY_TEMPLATES.get(category, [])
    catalog = list_scenarios().scenarios
    template_set = set(base_ids)
    out: list[str] = []
    for s in catalog:
        if "__p" not in s.id and s.id in template_set:
            out.append(s.id)
    for s in catalog:
        if "__p" in s.id and s.id.split("__")[0] in template_set:
            out.append(s.id)
    return out


# ---------------------------------------------------------------------------
# CSS — terminal styling per UI Build Addendum §3.2.
# ---------------------------------------------------------------------------


CSS = """
body, .gradio-container, button, input, select, textarea,
.cm-content, .cm-scroller, .cm-editor {
  font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', 'Consolas',
               ui-monospace, monospace !important;
}
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; }

#sre-header {
  display: flex; justify-content: space-between; align-items: flex-start;
  padding: 10px 4px 6px; border-bottom: 1px solid #30363d; margin-bottom: 6px;
}
#sre-title { color: #c9d1d9; font-weight: 700; font-size: 18px; letter-spacing: 1px; }
#sre-title small { color: #8b949e; font-weight: 400; font-size: 12px; display: block; margin-top: 2px; }
#sre-links { font-size: 12px; }
#sre-links a {
  color: #58a6ff; text-decoration: none; padding: 2px 8px;
  border: 1px solid #30363d; border-radius: 4px; margin-left: 6px;
  white-space: nowrap;
}
#sre-links a:hover { background: #161b22; }

#sre-banner {
  background: #161b22; border-left: 3px solid #d29922;
  padding: 8px 14px; margin: 6px 0 12px; color: #c9d1d9; font-size: 12px;
  border-radius: 4px;
}

.terminal textarea, .terminal pre, .terminal .cm-content,
.terminal .cm-scroller, .terminal .cm-editor {
  background: #0d1117 !important;
  color: #c9d1d9 !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}

.metric-bar {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  color: #c9d1d9; padding: 6px 0;
}

@media (max-width: 768px) {
  .terminal .cm-editor { max-height: 40vh !important; }
}
@media (min-width: 769px) {
  .terminal .cm-editor { max-height: 60vh !important; }
}
"""


HEADER_HTML = """
<div id="sre-header">
  <div id="sre-title">
    SRE-GYM <span style="color:#8b949e;font-weight:400">// tier-escalating SRE RL env</span>
    <small>RLVE incident-response environment. Each tier escalates a different
    dimension: compute → horizon → realism.</small>
  </div>
  <div id="sre-links">
    <a href="/docs" target="_blank" rel="noopener">API Docs</a>
    <a href="/mcp/tools" target="_blank" rel="noopener">MCP Tools</a>
    <a href="/info" target="_blank" rel="noopener">Legacy Info</a>
  </div>
</div>
"""


BANNER_HTML = """
<div id="sre-banner">
  Your tokens are held in this browser session only. They are never stored,
  logged, or sent anywhere except the provider you select.
</div>
"""


# ---------------------------------------------------------------------------
# Dropdown choice helpers.
# ---------------------------------------------------------------------------


def _model_choices(tier: Tier) -> list[tuple[str, str]]:
    """Return (label, value) tuples for the Model dropdown of a tier."""
    return [(e.label, e.label) for e in models_for_tier(tier)]


def _scenario_choices(tier: Tier, category: str | None = None) -> list[tuple[str, str]]:
    """Per-tier scenario dropdown choices.

    For Basic, ``category`` filters the templates+procgen-variants list.
    """
    if tier is Tier.BASIC:
        if category and category in CATEGORY_TEMPLATES:
            return [(sid, sid) for sid in _templates_in_category(category)]
        catalog = list_scenarios().scenarios
        base = [s.id for s in catalog if "__p" not in s.id]
        variants = [s.id for s in catalog if "__p" in s.id]
        return [(sid, sid) for sid in base + variants]
    if tier is Tier.ADVANCED:
        return [(sid, sid) for sid in list_advanced_scenarios()]
    if tier is Tier.MAX:
        return [(fid, fid) for fid in list_max_families()]
    return []


# ---------------------------------------------------------------------------
# Provider construction — wraps the address book in router.py.
# ---------------------------------------------------------------------------


def _resolve_provider(
    *,
    tier: Tier,
    model_label: str,
    custom_model_id: str,
    custom_base_url: str,
    hf_token: str,
    anthropic_key: str,
    openai_key: str,
    groq_key: str,
    together_key: str,
    fireworks_key: str,
    deepseek_key: str,
) -> tuple[Provider | None, str | None]:
    """Build the right Provider for the active tier + model selection.

    Returns ``(provider, error_message)``. When the user hasn't supplied a
    matching API key we return ``(None, msg)`` and the run handler falls
    back to ``DummyProvider`` (the trace still shows up — just deterministic).
    """
    entry = find_entry(model_label, tier) if model_label else None
    if entry is None and custom_model_id:
        # Custom model id without a curated match → assume HF inference unless
        # the caller also supplied a custom base_url, in which case treat as
        # OpenAI-compatible.
        if custom_base_url:
            entry = ModelEntry(
                label="custom",
                model_id=custom_model_id,
                kind=ProviderKind.OPENAI_COMPAT,
                base_url=custom_base_url,
                auth_key="openai_key",
                note="custom",
            )
        else:
            entry = ModelEntry(
                label="custom",
                model_id=custom_model_id,
                kind=ProviderKind.HF,
                note="custom",
            )
    if entry is None:
        return None, "no model selected — using offline DummyProvider"

    try:
        provider = build_provider(
            entry,
            hf_token=hf_token,
            anthropic_key=anthropic_key,
            openai_key=openai_key,
            groq_key=groq_key,
            together_key=together_key,
            fireworks_key=fireworks_key,
            deepseek_key=deepseek_key,
            custom_model_id=custom_model_id,
            custom_base_url=custom_base_url,
        )
        return provider, None
    except ProviderAuthError as exc:
        return None, str(exc)
    except SREGymError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Run handler — async generator that streams the cumulative trace.
# ---------------------------------------------------------------------------


def _local_base_url() -> str:
    """Pick the base URL to use for in-process /step calls."""
    return os.environ.get("SRE_GYM_BASE_URL", "http://127.0.0.1:7860")


async def run_handler(
    tier_value: str,
    category: str,
    scenario: str,
    chaos: str,
    seed: int,
    max_steps: int,
    model_label: str,
    custom_model_id: str,
    custom_base_url: str,
    hf_token: str,
    anthropic_key: str,
    openai_key: str,
    groq_key: str,
    together_key: str,
    fireworks_key: str,
    deepseek_key: str,
) -> AsyncIterator[tuple[str, str]]:
    """Stream a tier-aware episode trace into the terminal + metric bar.

    Yields (terminal_text, metric_markdown) on every step. The metric bar
    is updated only at episode start and end; intermediate streaming
    leaves the metric placeholder unchanged.
    """
    tier_str = (tier_value or "basic").lower()
    if tier_str not in {"basic", "advanced", "max"}:
        yield (f"[ERROR] unknown tier {tier_value!r}", "—")
        return
    tier = Tier(tier_str)

    provider, prov_err = _resolve_provider(
        tier=tier,
        model_label=model_label,
        custom_model_id=custom_model_id,
        custom_base_url=custom_base_url,
        hf_token=hf_token,
        anthropic_key=anthropic_key,
        openai_key=openai_key,
        groq_key=groq_key,
        together_key=together_key,
        fireworks_key=fireworks_key,
        deepseek_key=deepseek_key,
    )
    if provider is None:
        provider = DummyProvider()

    setup_lines: list[str] = []
    if prov_err:
        setup_lines.append(f"[provider] {prov_err}")
    if isinstance(provider, DummyProvider):
        setup_lines.append("[provider] using offline DummyProvider — paste an API key to drive a real model")

    metric_pending = "**reward:** —  |  **steps:** —  |  **resolved:** —  |  outcome=— valid=— fmt=— anti=— eff=—"
    if setup_lines:
        yield ("\n".join(setup_lines), metric_pending)

    last_text = "\n".join(setup_lines) if setup_lines else ""

    try:
        async for chunk in stream_episode(
            tier=tier_str,
            template=scenario if tier is Tier.BASIC else None,
            scenario=scenario if tier is Tier.ADVANCED else None,
            family=scenario if tier is Tier.MAX else None,
            chaos=chaos,
            seed=int(seed) if seed is not None else 0,
            provider=provider,
            base_url=_local_base_url(),
            max_steps=int(max_steps) if max_steps else 20,
            on_log=None,
        ):
            combined = (last_text + "\n" + chunk).strip() if last_text else chunk
            yield (combined, metric_pending)

        # Streaming complete — final metric line is parsed from the last chunk.
        last_block = combined if last_text else chunk  # type: ignore[possibly-undefined]
    except SREGymError as exc:
        yield ((last_text + f"\n[ERROR] {exc}").strip(), "—")
        return
    except Exception as exc:  # pragma: no cover - defensive
        yield ((last_text + f"\n[ERROR] {exc}").strip(), "—")
        return

    # Best-effort: pull the DONE / final reward line out of the final chunk
    # and render it as the metric bar.
    metric_md = _summarize_metric(last_block)
    yield (last_block, metric_md)


def _summarize_metric(trace_text: str) -> str:
    """Extract the final ``DONE …`` line and render the metric bar."""
    lines = [ln for ln in (trace_text or "").splitlines() if "DONE" in ln or "done" in ln]
    if not lines:
        return "**reward:** —  |  **steps:** —  |  **resolved:** —"
    summary = lines[-1]
    return f"<code>{summary}</code>"


# ---------------------------------------------------------------------------
# Tier-change wiring.
# ---------------------------------------------------------------------------


def on_tier_change(tier_value: str) -> tuple[Any, Any, Any, Any, Any]:
    """When the tier radio changes, repopulate the model + scenario dropdowns
    and toggle the Basic-only (category) and Max-only (chaos) controls."""
    try:
        tier = Tier((tier_value or "basic").lower())
    except ValueError:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    is_basic = tier is Tier.BASIC
    is_max = tier is Tier.MAX

    models = _model_choices(tier)
    default_model = models[0][1] if models else None

    if is_basic:
        cats = list(CATEGORY_TEMPLATES.keys())
        default_cat = cats[0]
        scenarios = _scenario_choices(tier, default_cat)
    else:
        scenarios = _scenario_choices(tier)
        default_cat = None

    default_scenario = scenarios[0][1] if scenarios else None

    return (
        gr.update(choices=models, value=default_model),
        gr.update(visible=is_basic, value=default_cat or "deploy"),
        gr.update(choices=scenarios, value=default_scenario),
        gr.update(visible=is_max),
        gr.update(visible=is_basic),    # max_steps slider only useful for Basic
    )


def on_category_change(tier_value: str, category: str) -> Any:
    """When the Basic-tier category changes, repopulate the template dropdown."""
    try:
        tier = Tier((tier_value or "basic").lower())
    except ValueError:
        return gr.update()
    if tier is not Tier.BASIC:
        return gr.update()
    scenarios = _scenario_choices(tier, category)
    default_scenario = scenarios[0][1] if scenarios else None
    return gr.update(choices=scenarios, value=default_scenario)


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app.
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    initial_tier = Tier.BASIC

    with gr.Blocks(title="sre-gym") as app:
        gr.HTML(HEADER_HTML)
        gr.HTML(BANNER_HTML)

        # gr.State holders for credentials. Never persisted. Never logged.
        hf_token_state = gr.State("")
        anthropic_state = gr.State("")
        openai_state = gr.State("")
        groq_state = gr.State("")
        together_state = gr.State("")
        fireworks_state = gr.State("")
        deepseek_state = gr.State("")

        with gr.Row(equal_height=True):
            # ---- Column A: TIER ----
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("**TIER**")
                tier_radio = gr.Radio(
                    choices=[("Basic", "basic"), ("Advanced", "advanced"), ("Max", "max")],
                    value=initial_tier.value,
                    label="",
                    interactive=True,
                )

            # ---- Column B: MODEL & API CONFIG ----
            with gr.Column(scale=2, min_width=420):
                gr.Markdown("**MODEL & API CONFIG**")
                model_dropdown = gr.Dropdown(
                    choices=_model_choices(initial_tier),
                    value=_model_choices(initial_tier)[0][1],
                    label="model",
                    interactive=True,
                    allow_custom_value=False,
                )
                with gr.Row():
                    hf_token_input = gr.Textbox(
                        label="HF token (hf_…)",
                        type="password",
                        placeholder="required for HF Inference models",
                        interactive=True,
                    )
                    anthropic_input = gr.Textbox(
                        label="Anthropic key",
                        type="password",
                        placeholder="sk-ant-…",
                        interactive=True,
                    )
                with gr.Row():
                    openai_input = gr.Textbox(
                        label="OpenAI key",
                        type="password",
                        placeholder="sk-…",
                        interactive=True,
                    )
                    groq_input = gr.Textbox(
                        label="Groq key",
                        type="password",
                        placeholder="gsk_…",
                        interactive=True,
                    )
                with gr.Row():
                    together_input = gr.Textbox(
                        label="Together key",
                        type="password",
                        interactive=True,
                    )
                    fireworks_input = gr.Textbox(
                        label="Fireworks key",
                        type="password",
                        placeholder="fw_…",
                        interactive=True,
                    )
                with gr.Row():
                    deepseek_input = gr.Textbox(
                        label="DeepSeek key",
                        type="password",
                        interactive=True,
                    )
                    custom_model_input = gr.Textbox(
                        label="custom model id (override)",
                        placeholder="e.g. mistralai/Mistral-Small-Instruct",
                        interactive=True,
                    )
                custom_base_input = gr.Textbox(
                    label="custom base_url (only for OpenAI-compatible providers)",
                    placeholder="https://api.example.com/v1",
                    interactive=True,
                )

            # ---- Column C: SCENARIO ----
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("**SCENARIO**")
                category_dropdown = gr.Dropdown(
                    choices=list(CATEGORY_TEMPLATES.keys()),
                    value="deploy",
                    label="category (Basic only)",
                    interactive=True,
                    visible=True,
                )
                scenario_dropdown = gr.Dropdown(
                    choices=_scenario_choices(initial_tier, "deploy"),
                    value=_scenario_choices(initial_tier, "deploy")[0][1],
                    label="template / scenario / family",
                    interactive=True,
                    allow_custom_value=True,
                )
                chaos_dropdown = gr.Dropdown(
                    choices=list(CHAOS_PATTERNS),
                    value=list(CHAOS_PATTERNS)[0],
                    label="chaos pattern (Max only)",
                    interactive=True,
                    visible=False,
                )
                seed_input = gr.Number(value=42, label="seed", precision=0, interactive=True)
                max_steps_slider = gr.Slider(
                    minimum=5, maximum=50, value=20, step=1,
                    label="max steps", interactive=True,
                )

        # ---- Terminal ----
        terminal = gr.Code(
            label="rollout terminal",
            language="shell",
            value=(
                "$ sre-gym ready. configure above and click ▶ run.\n"
                "$ note: Basic & Max default to scripted-optimal if no API key is set;\n"
                "$       Advanced needs a model for a real long-horizon trace.\n"
            ),
            interactive=False,
            lines=24,
            elem_classes=["terminal"],
        )
        metrics = gr.Markdown(
            "**reward:** —  |  **steps:** —  |  **resolved:** —  |  outcome=— valid=— fmt=— anti=— eff=—",
            elem_classes=["metric-bar"],
        )

        with gr.Row():
            run_btn = gr.Button("▶  run", variant="primary")
            stop_btn = gr.Button("■  stop")
            reset_btn = gr.Button("↻  reset")

        # ---- Event wiring ----

        # Sync API-key inputs into gr.State (never persisted server-side).
        hf_token_input.change(lambda x: x, inputs=[hf_token_input], outputs=[hf_token_state])
        anthropic_input.change(lambda x: x, inputs=[anthropic_input], outputs=[anthropic_state])
        openai_input.change(lambda x: x, inputs=[openai_input], outputs=[openai_state])
        groq_input.change(lambda x: x, inputs=[groq_input], outputs=[groq_state])
        together_input.change(lambda x: x, inputs=[together_input], outputs=[together_state])
        fireworks_input.change(lambda x: x, inputs=[fireworks_input], outputs=[fireworks_state])
        deepseek_input.change(lambda x: x, inputs=[deepseek_input], outputs=[deepseek_state])

        tier_radio.change(
            on_tier_change,
            inputs=[tier_radio],
            outputs=[model_dropdown, category_dropdown, scenario_dropdown, chaos_dropdown, max_steps_slider],
        )
        category_dropdown.change(
            on_category_change,
            inputs=[tier_radio, category_dropdown],
            outputs=[scenario_dropdown],
        )

        run_event = run_btn.click(
            run_handler,
            inputs=[
                tier_radio, category_dropdown, scenario_dropdown, chaos_dropdown,
                seed_input, max_steps_slider,
                model_dropdown, custom_model_input, custom_base_input,
                hf_token_state, anthropic_state, openai_state, groq_state,
                together_state, fireworks_state, deepseek_state,
            ],
            outputs=[terminal, metrics],
        )
        stop_btn.click(None, None, None, cancels=[run_event])
        reset_btn.click(
            lambda: (
                "$ terminal cleared.\n",
                "**reward:** —  |  **steps:** —  |  **resolved:** —",
            ),
            inputs=None,
            outputs=[terminal, metrics],
        )

        gr.Markdown(
            "<small style='color:#8b949e'>"
            "sre-gym v3.0 · 12 Basic templates × 5 procgen = 72 scenarios · "
            "3 Advanced reference scenarios · 1 Max family with 11 chaos patterns. "
            "<a href='https://github.com/dakshdoesdev/sre-enginnerllm' target='_blank' rel='noopener' style='color:#58a6ff'>github</a> · "
            "<a href='https://huggingface.co/spaces/Madhav189/sre-env' target='_blank' rel='noopener' style='color:#58a6ff'>HF Space</a>"
            "</small>"
        )

    return app


# ---------------------------------------------------------------------------
# Mount Gradio onto the existing FastAPI app.
# ---------------------------------------------------------------------------


def _build_combined_app() -> Any:
    """Mount the Gradio Blocks UI at the root of the existing FastAPI server.

    Per the UI Build Addendum §1, ``/`` serves Gradio and every existing
    FastAPI route (``/step`` ``/reset`` ``/state`` ``/tasks`` ``/baseline``
    ``/grader`` ``/status`` ``/health`` ``/metadata`` ``/schema`` ``/info``
    ``/simple`` ``/docs`` ``/redoc`` ``/openapi.json`` ``/mcp`` ``/mcp/tools``
    ``/mcp/reset``) is preserved untouched.

    Implementation: ``gradio.routes.mount_gradio_app(api_app, blocks, path="/")``
    is the canonical way to attach a Gradio Blocks instance to an existing
    FastAPI app — it registers Gradio's websocket + asset handlers without
    touching pre-existing routes.
    """
    from gradio.routes import mount_gradio_app
    from unified_incident_env.server.app import create_compatible_app as create_env_app

    blocks = build_app()
    blocks.queue(default_concurrency_limit=4)
    blocks.css = CSS
    api_app = create_env_app()
    return mount_gradio_app(api_app, blocks, path="/")


def main() -> None:
    """Local launcher. Production uses ``uvicorn app:app`` — see Dockerfile."""
    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    import uvicorn

    uvicorn.run("app:app", host=host, port=server_port, log_level="info")


# Module-level FastAPI app — exposes ``app:app`` for ``uvicorn app:app``
# (matches the Dockerfile CMD exactly).
app = _build_combined_app()


if __name__ == "__main__":
    main()
