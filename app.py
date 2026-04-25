"""Terminal-style Gradio UI for sre-gym.

Three tiers are exposed through a single page:

- Basic — runs against the in-process UnifiedIncidentEnvironment (12 templates × 5 procgen).
- Advanced — chains Basic episodes via sre_gym.advanced.runner with horizon state.
- Max — runs the Python state-machine simulator over the 22-node service graph.

Tokens (HF, Anthropic, OpenAI, Groq, Together, Fireworks, DeepSeek) are held
ONLY in gr.State for the lifetime of the user's session. They are never
written to disk, never logged, and never echoed back through the UI.

This file is intentionally rendered as a single launcher; supporting modules
live in ``sre_gym/ui/{providers,router,policies}.py``.

Run locally::

    python app.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Iterator

import gradio as gr

from sre_gym.advanced.runner import (
    list_advanced_scenarios,
    run_advanced,
)
from sre_gym.basic_runner import run_basic
from sre_gym.exceptions import (
    ProviderAuthError,
    SREGymError,
)
from sre_gym.max.runner import (
    CHAOS_PATTERNS,
    list_max_families,
    run_max,
)
from sre_gym.tier import Tier
from sre_gym.ui.policies import make_policy
from sre_gym.ui.router import (
    ModelEntry,
    ProviderKind,
    build_provider,
    find_entry,
    models_for_tier,
)
from unified_incident_env.server.challenge import list_scenarios

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Terminal styling.
# ---------------------------------------------------------------------------


CSS = """
body, .gradio-container {
  font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', 'Consolas', monospace !important;
}
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
}
#sre-banner {
  background: #161b22;
  border-left: 3px solid #58a6ff;
  padding: 10px 14px;
  margin-bottom: 8px;
  color: #c9d1d9;
  font-size: 13px;
  border-radius: 4px;
}
#sre-title {
  color: #c9d1d9;
  font-weight: 700;
  font-size: 18px;
  letter-spacing: 1px;
  padding: 6px 0;
}
#sre-title span.dim { color: #8b949e; font-weight: 400; }
.terminal textarea, .terminal pre, .terminal .cm-content {
  background: #0d1117 !important;
  color: #c9d1d9 !important;
  font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
.terminal-line { color: #c9d1d9; }
.metric-bar {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: #c9d1d9;
  padding: 6px 0;
}
.security-tag { color: #f85149; font-weight: 600; }
input, select, textarea, .cm-content { font-family: 'JetBrains Mono', monospace !important; }
"""


HEADER = """
<div id="sre-title">
  SRE-GYM <span class="dim">// tier-escalating incident-response RL env</span>
</div>
<div id="sre-banner">
  Each tier escalates a different dimension: <b>Basic</b> escalates compute,
  <b>Advanced</b> escalates horizon, <b>Max</b> escalates realism.
  Your tokens are held in-session only — never stored, logged, or sent anywhere
  except the provider you selected.
</div>
"""


# ---------------------------------------------------------------------------
# Helpers — terminal output formatting.
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.utcnow().strftime("%M:%S")


def _line(text: str) -> str:
    return f"[{_ts()}] {text}"


def _truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _scenario_choices_for_tier(tier: Tier) -> list[tuple[str, str]]:
    """Return (label, value) pairs for the scenario dropdown."""
    if tier is Tier.BASIC:
        catalog = list_scenarios().scenarios
        # Group: show base templates first, then procgen variants.
        base = [s for s in catalog if "__p" not in s.id]
        variants = [s for s in catalog if "__p" in s.id]
        return [(s.id, s.id) for s in base + variants]
    if tier is Tier.ADVANCED:
        return [(sid, sid) for sid in list_advanced_scenarios()]
    if tier is Tier.MAX:
        return [(fid, fid) for fid in list_max_families()]
    return []


def _model_choices_for_tier(tier: Tier) -> list[tuple[str, str]]:
    return [(entry.label, entry.label) for entry in models_for_tier(tier)]


# ---------------------------------------------------------------------------
# Run dispatch — generator that yields cumulative terminal text per step.
# ---------------------------------------------------------------------------


def _run_basic_stream(
    scenario_id: str,
    seed: int,
    policy_fn: Any,
) -> Iterator[tuple[str, str]]:
    """Run a Basic episode, yielding (terminal_text, metrics_text) tuples."""
    buffer: list[str] = []

    def on_log(line: str) -> None:
        buffer.append(_line(line))

    on_log(f"=== sre-gym Basic :: scenario={scenario_id} seed={seed} ===")
    yield "\n".join(buffer), "reward: —  steps: —  resolved: —"

    result = run_basic(scenario_id, seed=seed, policy=policy_fn, on_log=on_log)

    # After completion, emit final summary.
    on_log(f"DONE  reward={result.final_score:.3f}  resolved={result.incident_resolved}  steps={result.tick_count}")
    metrics = f"reward: {result.final_score:.3f}  steps: {result.tick_count}  resolved: {result.incident_resolved}"
    yield "\n".join(buffer), metrics


def _run_advanced_stream(
    scenario_id: str,
    seed: int,
    policy_fn: Any,
) -> Iterator[tuple[str, str]]:
    buffer: list[str] = []

    def on_log(line: str) -> None:
        buffer.append(_line(line))

    on_log(f"=== sre-gym Advanced :: scenario={scenario_id} seed={seed} ===")
    yield "\n".join(buffer), "reward: —  phases: —  resolved: —"

    result = run_advanced(scenario_id, seed=seed, policy=policy_fn, on_log=on_log)

    on_log(
        f"DONE  reward={result.final_reward:.3f}  decay×{result.horizon_decay_factor:.3f}  "
        f"phases={len(result.phases)}  success={result.success}"
    )
    metrics = (
        f"reward: {result.final_reward:.3f}  "
        f"phases: {len(result.phases)}  success: {result.success}"
    )
    yield "\n".join(buffer), metrics


def _run_max_stream(
    family_id: str,
    chaos: str,
    seed: int,
    policy_fn: Any,
) -> Iterator[tuple[str, str]]:
    buffer: list[str] = []

    def on_log(line: str) -> None:
        buffer.append(_line(line))

    on_log(f"=== sre-gym Max :: family={family_id} chaos={chaos} seed={seed} ===")
    yield "\n".join(buffer), "reward: —  steps: —  resolved: —"

    result = run_max(family_id, chaos=chaos, seed=seed, policy=policy_fn, on_log=on_log)

    sec_tag = ""
    if result.classification == "security":
        sec_tag = "  [SECURITY]"
    on_log(
        f"DONE{sec_tag}  reward={result.final_reward:.3f}  resolved={result.incident_resolved}  "
        f"blast={result.blast_radius}"
    )
    metrics = (
        f"reward: {result.final_reward:.3f}  steps: {result.tick_count}  "
        f"resolved: {result.incident_resolved}  blast: {result.blast_radius}"
    )
    yield "\n".join(buffer), metrics


# ---------------------------------------------------------------------------
# Top-level run handler.
# ---------------------------------------------------------------------------


def _resolve_policy(
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
    on_log: Any,
) -> Any:
    """Return a policy function or None (None == use the runner's scripted default)."""
    entry = find_entry(model_label, tier) if model_label else None
    if entry is None and not custom_model_id:
        # No model picked → run the scripted-optimal baseline.
        return None
    if entry is None:
        # Custom model ID without a curated entry — assume HF inference.
        entry = ModelEntry(
            label="custom",
            model_id=custom_model_id,
            kind=ProviderKind.HF,
            note="custom",
        )
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
    except ProviderAuthError as exc:
        on_log(_line(f"[provider] {exc}"))
        return None
    except SREGymError as exc:
        on_log(_line(f"[provider] {exc}"))
        return None

    return make_policy(provider, tier=tier.value, on_log=lambda s: on_log(_line(s)))


def run_handler(
    tier_value: str,
    scenario: str,
    chaos: str,
    seed: int,
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
) -> Iterator[tuple[str, str]]:
    """Main run button handler — generator that streams terminal output."""
    try:
        tier = Tier(tier_value)
    except ValueError:
        yield f"[{_ts()}] error: unknown tier {tier_value!r}", "—"
        return

    log_buffer: list[str] = []

    def emit(line: str) -> None:
        log_buffer.append(line)

    policy = _resolve_policy(
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
        on_log=emit,
    )
    if policy is None:
        emit(_line("[policy] no API key supplied or model not configured — using scripted-optimal baseline"))

    # Yield any setup-time log lines first.
    if log_buffer:
        yield "\n".join(log_buffer), "reward: —  steps: —  resolved: —"

    try:
        if tier is Tier.BASIC:
            for terminal, metrics in _run_basic_stream(scenario, seed, policy):
                yield "\n".join(log_buffer + [terminal]) if log_buffer else terminal, metrics
        elif tier is Tier.ADVANCED:
            for terminal, metrics in _run_advanced_stream(scenario, seed, policy):
                yield "\n".join(log_buffer + [terminal]) if log_buffer else terminal, metrics
        elif tier is Tier.MAX:
            for terminal, metrics in _run_max_stream(scenario, chaos, seed, policy):
                yield "\n".join(log_buffer + [terminal]) if log_buffer else terminal, metrics
        else:  # pragma: no cover
            yield "\n".join(log_buffer + [_line(f"unknown tier {tier_value!r}")]), "—"
    except SREGymError as exc:
        emit(_line(f"error: {exc}"))
        yield "\n".join(log_buffer), "—"


# ---------------------------------------------------------------------------
# Tier-change wiring.
# ---------------------------------------------------------------------------


def on_tier_change(tier_value: str) -> tuple[Any, Any, Any]:
    """When tier changes, repopulate the model + scenario dropdowns."""
    try:
        tier = Tier(tier_value)
    except ValueError:
        return gr.update(), gr.update(), gr.update()

    model_choices = _model_choices_for_tier(tier)
    scenario_choices = _scenario_choices_for_tier(tier)

    default_model = model_choices[0][1] if model_choices else None
    default_scenario = scenario_choices[0][1] if scenario_choices else None

    chaos_visible = tier is Tier.MAX
    return (
        gr.update(choices=model_choices, value=default_model),
        gr.update(choices=scenario_choices, value=default_scenario),
        gr.update(visible=chaos_visible),
    )


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app.
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    initial_tier = Tier.BASIC

    with gr.Blocks(title="sre-gym") as app:
        gr.HTML(HEADER)

        # gr.State holders for credentials. Never persisted. Never logged.
        hf_token_state = gr.State("")
        anthropic_state = gr.State("")
        openai_state = gr.State("")
        groq_state = gr.State("")
        together_state = gr.State("")
        fireworks_state = gr.State("")
        deepseek_state = gr.State("")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("**TIER**")
                tier_radio = gr.Radio(
                    choices=[("Basic", Tier.BASIC.value), ("Advanced", Tier.ADVANCED.value), ("Max", Tier.MAX.value)],
                    value=initial_tier.value,
                    label="",
                    interactive=True,
                )
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("**MODEL & API CONFIG**")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=_model_choices_for_tier(initial_tier),
                        value=_model_choices_for_tier(initial_tier)[0][1],
                        label="model",
                        interactive=True,
                        allow_custom_value=False,
                    )
                with gr.Row():
                    hf_token_input = gr.Textbox(
                        label="HF token",
                        type="password",
                        placeholder="hf_…",
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
                        placeholder="…",
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
                        placeholder="sk-…",
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
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("**SCENARIO**")
                scenario_dropdown = gr.Dropdown(
                    choices=_scenario_choices_for_tier(initial_tier),
                    value=_scenario_choices_for_tier(initial_tier)[0][1] if _scenario_choices_for_tier(initial_tier) else None,
                    label="scenario / family",
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

        terminal = gr.Code(
            label="rollout terminal",
            language="shell",
            value=("[ready] pick a tier, model, and scenario — then press ▶ run.\n"
                   "[note]  basic & max default to scripted-optimal if no API key is set.\n"
                   "[note]  advanced needs at minimum the scripted policy to demonstrate "
                   "horizon-decay; supply a model for a real run."),
            interactive=False,
            lines=24,
            elem_classes=["terminal"],
        )
        metrics = gr.Markdown("reward: —  steps: —  resolved: —", elem_classes=["metric-bar"])

        with gr.Row():
            run_btn = gr.Button("▶  run", variant="primary")
            reset_btn = gr.Button("↻  reset")
            stop_btn = gr.Button("■  stop")

        # ------ Event wiring ------

        # Sync token input -> gr.State (never stored on disk).
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
            outputs=[model_dropdown, scenario_dropdown, chaos_dropdown],
        )

        run_event = run_btn.click(
            run_handler,
            inputs=[
                tier_radio, scenario_dropdown, chaos_dropdown, seed_input,
                model_dropdown, custom_model_input, custom_base_input,
                hf_token_state, anthropic_state, openai_state, groq_state,
                together_state, fireworks_state, deepseek_state,
            ],
            outputs=[terminal, metrics],
        )
        stop_btn.click(None, None, None, cancels=[run_event])
        reset_btn.click(
            lambda: ("[reset] terminal cleared.\n", "reward: —  steps: —  resolved: —"),
            inputs=None,
            outputs=[terminal, metrics],
        )

        gr.Markdown(
            "<small style='color:#8b949e'>"
            "sre-gym v3.0 · 12 templates × 5 procgen = 72 Basic scenarios · "
            "3 Advanced reference scenarios · 1 Max family with 11 chaos patterns. "
            "<a href='https://github.com/dakshdoesdev/sre-enginnerllm' style='color:#58a6ff'>github</a> · "
            "<a href='https://huggingface.co/spaces/Madhav189/sre-env' style='color:#58a6ff'>HF Space</a>"
            "</small>"
        )

    return app


def _build_combined_app() -> Any:
    """Mount the Gradio Blocks UI inside the existing FastAPI server.

    This is the trick that lets the HF Space expose:
      - ``/``                     Gradio terminal UI
      - ``/api/health``           OpenEnv health probe
      - ``/api/tasks``            72-scenario catalogue
      - ``/api/reset`` ``/api/step`` ``/api/state`` — OpenEnv contract
      - ``/api/mcp``              JSON-RPC 2.0 MCP dual-route
      - ``/api/baseline`` ``/api/grader`` ``/api/status``
    on a single port (7860 by default), so HF Space's single ``app_port``
    is sufficient for both the UI and the env API.
    """
    import gradio as gr  # local re-import for type-checker
    from unified_incident_env.server.app import create_compatible_app as create_env_app

    blocks = build_app()
    blocks.queue(default_concurrency_limit=4)
    blocks.css = CSS
    fastapi_app = create_env_app()
    # Mount Gradio at /, FastAPI env routes stay under their original paths.
    # Gradio's mount_gradio_app re-uses the FastAPI server, so /tasks /step
    # etc. remain reachable; we ALSO alias them under /api for clarity.
    return gr.mount_gradio_app(fastapi_app, blocks, path="/ui")


def main() -> None:
    server_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    if os.environ.get("SRE_GYM_UI_ONLY") == "1":
        # Standalone Gradio (no env API) — keep `python app.py` working without
        # the openenv stack for users who only want the UI demo.
        blocks = build_app()
        blocks.queue(default_concurrency_limit=4).launch(
            server_name=host,
            server_port=server_port,
            show_api=False,
            share=False,
            css=CSS,
            theme=gr.themes.Base(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
            ),
        )
        return

    # Combined: Gradio UI mounted inside the OpenEnv FastAPI server.
    import uvicorn
    combined = _build_combined_app()
    uvicorn.run(combined, host=host, port=server_port, log_level="info")


if __name__ == "__main__":
    main()
