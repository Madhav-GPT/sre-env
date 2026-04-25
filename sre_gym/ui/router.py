"""Per-tier curated model lists + provider router.

The Gradio UI renders a model dropdown that auto-populates from the active
tier's curated list. The user can also override with any HF model ID or
custom OpenAI-compatible base URL — see ``app.py`` for the BYOK panel.

Model IDs are validated against published catalogues as of late 2025 / early
2026. Anything we can't verify is documented as ``placeholder`` so users
don't waste API spend chasing nonexistent models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sre_gym.exceptions import ProviderAuthError
from sre_gym.tier import Tier
from sre_gym.ui.providers import (
    AnthropicProvider,
    HFInferenceProvider,
    OpenAICompatibleProvider,
    Provider,
)


class ProviderKind(str, Enum):
    HF = "hf"
    ANTHROPIC = "anthropic"
    OPENAI_COMPAT = "openai-compat"


@dataclass(frozen=True)
class ModelEntry:
    """One curated model entry shown in the UI dropdown."""

    label: str            # human-friendly name shown in the dropdown
    model_id: str         # provider-side model identifier
    kind: ProviderKind
    base_url: Optional[str] = None      # only set for OPENAI_COMPAT
    auth_key: str = "hf_token"          # which gr.State holds the credential
    note: str = ""                      # surfaced as helper text in the UI


# ---------------------------------------------------------------------------
# Curated model lists, by tier.
#
# All model IDs are real (verified at submission time, 2026-04). Where
# something is hypothetical, the entry is omitted rather than ship a 404.
# ---------------------------------------------------------------------------


BASIC_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        label="trained 3B (sre-gym-qwen25-3b-grpo)",
        model_id="dakshdoesdev/sre-gym-qwen25-3b-grpo",
        kind=ProviderKind.HF,
        note="default — the trained specialist (LoRA over Qwen2.5-3B-Instruct)",
    ),
    ModelEntry(
        label="Qwen2.5-7B-Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        kind=ProviderKind.HF,
        note="open-weight, runs anywhere with HF inference",
    ),
    ModelEntry(
        label="Llama-3.1-8B-Instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        kind=ProviderKind.HF,
    ),
    ModelEntry(
        label="Gemma-2-27B-IT",
        model_id="google/gemma-2-27b-it",
        kind=ProviderKind.HF,
    ),
    ModelEntry(
        label="Claude Sonnet 4.6 (BYOK)",
        model_id="claude-sonnet-4-6",
        kind=ProviderKind.ANTHROPIC,
        auth_key="anthropic_key",
    ),
    ModelEntry(
        label="GPT-5 (BYOK)",
        model_id="gpt-5",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.openai.com/v1",
        auth_key="openai_key",
    ),
    ModelEntry(
        label="Llama-3.3-70B Together (BYOK)",
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.together.xyz/v1",
        auth_key="together_key",
    ),
)


ADVANCED_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        label="Llama-3.3-70B Together (BYOK)",
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.together.xyz/v1",
        auth_key="together_key",
        note="default — strong long-horizon coherence at moderate cost",
    ),
    ModelEntry(
        label="Claude Sonnet 4.6 (BYOK)",
        model_id="claude-sonnet-4-6",
        kind=ProviderKind.ANTHROPIC,
        auth_key="anthropic_key",
    ),
    ModelEntry(
        label="GLM-4.6 (HF)",
        model_id="zai-org/GLM-4.6",
        kind=ProviderKind.HF,
        note="long-context Chinese open model",
    ),
    ModelEntry(
        label="Kimi-K2-Instruct (HF)",
        model_id="moonshotai/Kimi-K2-Instruct",
        kind=ProviderKind.HF,
    ),
    ModelEntry(
        label="Llama-3.3-70B Fireworks (BYOK)",
        model_id="accounts/fireworks/models/llama-v3p3-70b-instruct",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.fireworks.ai/inference/v1",
        auth_key="fireworks_key",
    ),
    ModelEntry(
        label="Llama-3.3-70B Groq (BYOK)",
        model_id="llama-3.3-70b-versatile",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.groq.com/openai/v1",
        auth_key="groq_key",
        note="free tier, 14k req/day",
    ),
)


MAX_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        label="Claude Sonnet 4.6 (BYOK)",
        model_id="claude-sonnet-4-6",
        kind=ProviderKind.ANTHROPIC,
        auth_key="anthropic_key",
        note="default — strongest realism-tier reasoning",
    ),
    ModelEntry(
        label="Claude Opus 4.7 (BYOK)",
        model_id="claude-opus-4-7",
        kind=ProviderKind.ANTHROPIC,
        auth_key="anthropic_key",
    ),
    ModelEntry(
        label="GPT-5 (BYOK)",
        model_id="gpt-5",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.openai.com/v1",
        auth_key="openai_key",
    ),
    ModelEntry(
        label="GLM-4.6 (HF)",
        model_id="zai-org/GLM-4.6",
        kind=ProviderKind.HF,
    ),
    ModelEntry(
        label="DeepSeek-V3 (BYOK)",
        model_id="deepseek-chat",
        kind=ProviderKind.OPENAI_COMPAT,
        base_url="https://api.deepseek.com/v1",
        auth_key="deepseek_key",
    ),
)


TIER_MODELS: dict[Tier, tuple[ModelEntry, ...]] = {
    Tier.BASIC: BASIC_MODELS,
    Tier.ADVANCED: ADVANCED_MODELS,
    Tier.MAX: MAX_MODELS,
}


def models_for_tier(tier: Tier) -> list[ModelEntry]:
    return list(TIER_MODELS[tier])


def find_entry(label: str, tier: Tier) -> Optional[ModelEntry]:
    for entry in TIER_MODELS[tier]:
        if entry.label == label or entry.model_id == label:
            return entry
    return None


# ---------------------------------------------------------------------------
# Provider construction.
# ---------------------------------------------------------------------------


def build_provider(
    entry: ModelEntry,
    *,
    hf_token: str = "",
    anthropic_key: str = "",
    openai_key: str = "",
    groq_key: str = "",
    together_key: str = "",
    fireworks_key: str = "",
    deepseek_key: str = "",
    custom_model_id: str = "",
    custom_base_url: str = "",
) -> Provider:
    """Construct a Provider for the given ModelEntry + user-supplied credentials.

    The ``custom_model_id`` and ``custom_base_url`` arguments override the
    entry's defaults — that's the BYOK escape hatch for users who want to
    point at any HF model or any OpenAI-compatible endpoint not on the
    curated list.
    """
    model_id = custom_model_id or entry.model_id
    auth_keys = {
        "hf_token": hf_token,
        "anthropic_key": anthropic_key,
        "openai_key": openai_key,
        "groq_key": groq_key,
        "together_key": together_key,
        "fireworks_key": fireworks_key,
        "deepseek_key": deepseek_key,
    }

    if entry.kind is ProviderKind.HF:
        token = auth_keys["hf_token"]
        if not token:
            raise ProviderAuthError("hf", "HF token required for this model — paste it in the API config panel")
        return HFInferenceProvider(hf_token=token, model=model_id)

    if entry.kind is ProviderKind.ANTHROPIC:
        key = auth_keys["anthropic_key"]
        if not key:
            raise ProviderAuthError("anthropic", "Anthropic API key required for this model")
        return AnthropicProvider(api_key=key, model=model_id)

    if entry.kind is ProviderKind.OPENAI_COMPAT:
        key = auth_keys.get(entry.auth_key, "")
        if not key:
            raise ProviderAuthError(
                entry.auth_key.replace("_key", ""),
                f"API key required ({entry.auth_key})",
            )
        base = custom_base_url or entry.base_url or ""
        return OpenAICompatibleProvider(base_url=base, api_key=key, model=model_id)

    raise ValueError(f"unknown provider kind {entry.kind!r}")
