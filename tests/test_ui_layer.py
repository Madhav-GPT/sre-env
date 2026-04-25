"""Tests for the BYOK provider routing + Gradio UI helpers.

We don't run the full Gradio server here (that's an integration test); these
tests exercise the building blocks app.py wires together:

- ``sre_gym/ui/router.py``    — model lists, find_entry, build_provider
- ``sre_gym/ui/providers.py`` — Provider auth checks + exception mapping
- ``sre_gym/ui/policies.py``  — JSON action extraction + fallback to escalate
"""

from __future__ import annotations

from typing import Any

import pytest

from sre_gym.exceptions import ActionParseError, ProviderAuthError, ProviderModelError
from sre_gym.tier import Tier
from sre_gym.ui.policies import _extract_json_object, make_policy
from sre_gym.ui.providers import (
    AnthropicProvider,
    HFInferenceProvider,
    OpenAICompatibleProvider,
)
from sre_gym.ui.router import (
    ADVANCED_MODELS,
    BASIC_MODELS,
    MAX_MODELS,
    ProviderKind,
    build_provider,
    find_entry,
    models_for_tier,
)


# ---------- Router / model catalogue ----------


def test_models_for_tier_returns_curated_list_for_each_tier() -> None:
    for tier in Tier:
        models = models_for_tier(tier)
        assert len(models) >= 3, f"{tier.value} must have at least 3 curated models"
        # Each entry must have a label, model_id, and provider kind.
        for entry in models:
            assert entry.label
            assert entry.model_id
            assert isinstance(entry.kind, ProviderKind)


def test_basic_tier_default_is_trained_3b() -> None:
    """The Basic tier's first entry should be the trained specialist."""
    assert BASIC_MODELS[0].model_id == "dakshdoesdev/sre-gym-qwen25-3b-grpo"


def test_advanced_tier_default_is_long_horizon_model() -> None:
    """The Advanced tier's first entry must be a long-horizon-class model."""
    assert "70B" in ADVANCED_MODELS[0].model_id or "Llama-3.3-70B" in ADVANCED_MODELS[0].label


def test_max_tier_default_is_claude_sonnet() -> None:
    """The Max tier's first entry must be Claude Sonnet (BYOK)."""
    assert "Sonnet" in MAX_MODELS[0].label
    assert MAX_MODELS[0].kind is ProviderKind.ANTHROPIC


def test_find_entry_resolves_label_or_model_id() -> None:
    entry = find_entry(BASIC_MODELS[1].label, Tier.BASIC)
    assert entry is BASIC_MODELS[1]
    entry = find_entry(BASIC_MODELS[1].model_id, Tier.BASIC)
    assert entry is BASIC_MODELS[1]
    assert find_entry("nonexistent", Tier.BASIC) is None


# ---------- Providers — auth + exception mapping ----------


def test_hf_provider_rejects_empty_token() -> None:
    with pytest.raises(ProviderAuthError):
        HFInferenceProvider(hf_token="", model="Qwen/Qwen2.5-7B-Instruct")


def test_anthropic_provider_rejects_empty_key() -> None:
    with pytest.raises(ProviderAuthError):
        AnthropicProvider(api_key="", model="claude-sonnet-4-6")


def test_openai_compat_provider_rejects_empty_key() -> None:
    with pytest.raises(ProviderAuthError):
        OpenAICompatibleProvider(base_url="https://api.openai.com/v1", api_key="", model="gpt-5")


def test_openai_compat_provider_rejects_empty_base_url() -> None:
    with pytest.raises(ProviderModelError):
        OpenAICompatibleProvider(base_url="", api_key="sk-test", model="gpt-5")


# ---------- build_provider — credential dispatch ----------


def test_build_provider_for_basic_default_uses_hf_token() -> None:
    """Building the trained-3B provider with an HF token works."""
    entry = BASIC_MODELS[0]
    provider = build_provider(entry, hf_token="hf_test")
    assert isinstance(provider, HFInferenceProvider)
    assert provider.model == entry.model_id


def test_build_provider_anthropic_default() -> None:
    sonnet = next(e for e in MAX_MODELS if "Sonnet" in e.label)
    provider = build_provider(sonnet, anthropic_key="sk-ant-test")
    assert isinstance(provider, AnthropicProvider)


def test_build_provider_openai_compat_default() -> None:
    gpt5 = next(e for e in MAX_MODELS if "GPT-5" in e.label)
    provider = build_provider(gpt5, openai_key="sk-test")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "gpt-5"


def test_build_provider_missing_credential_raises_provider_auth_error() -> None:
    """No HF token → should raise ProviderAuthError, not silently proceed."""
    entry = BASIC_MODELS[0]
    with pytest.raises(ProviderAuthError):
        build_provider(entry, hf_token="")


def test_build_provider_custom_model_override() -> None:
    """``custom_model_id`` overrides the entry's model_id."""
    entry = BASIC_MODELS[0]
    provider = build_provider(entry, hf_token="hf_test", custom_model_id="mistralai/Mistral-Small-Instruct")
    assert provider.model == "mistralai/Mistral-Small-Instruct"


# ---------- Policy adapter — JSON extraction + fallbacks ----------


def test_extract_json_object_strips_markdown_fences() -> None:
    text = '```json\n{"action_type": "query_logs", "service": "worker"}\n```'
    obj = _extract_json_object(text)
    assert obj == {"action_type": "query_logs", "service": "worker"}


def test_extract_json_object_handles_prose_around_json() -> None:
    text = 'Sure thing! Here is the action:\n{"action_type": "escalate"}\nLet me know if you need more.'
    obj = _extract_json_object(text)
    assert obj == {"action_type": "escalate"}


def test_extract_json_object_normalizes_action_alias() -> None:
    text = '{"action": "query_logs", "service": "worker"}'
    obj = _extract_json_object(text)
    assert obj == {"action_type": "query_logs", "service": "worker"}


def test_extract_json_object_raises_on_unterminated() -> None:
    with pytest.raises(ActionParseError):
        _extract_json_object('{"action_type": "query_logs"')


def test_extract_json_object_raises_on_no_json() -> None:
    with pytest.raises(ActionParseError):
        _extract_json_object('I am sorry but I cannot respond as JSON.')


class _FakeProvider:
    """Stand-in for tests — captures the messages and returns a configured response."""

    name = "fake"
    model = "fake-model"

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_messages = messages
        return self._response

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # pragma: no cover
        return self.chat_sync(messages, **kwargs)


def test_make_policy_returns_action_dict_for_well_formed_response() -> None:
    fake = _FakeProvider('{"action_type":"query_logs","service":"worker"}')
    policy = make_policy(fake, tier="basic")

    class FakeObs:
        prompt_text = "incident summary text"

    action = policy(FakeObs())
    assert action == {"action_type": "query_logs", "service": "worker"}
    # System prompt + user observation must have been sent.
    assert fake.last_messages[0]["role"] == "system"
    assert fake.last_messages[1]["role"] == "user"


def test_make_policy_falls_back_to_escalate_on_garbage() -> None:
    fake = _FakeProvider("not json at all, sorry")
    policy = make_policy(fake, tier="basic")

    class FakeObs:
        prompt_text = "..."

    action = policy(FakeObs())
    assert action == {"action_type": "escalate"}


def test_make_policy_falls_back_to_escalate_on_provider_auth_error() -> None:
    class FakeAuthBrokenProvider:
        name = "anthropic"
        model = "claude-sonnet-4-6"

        def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            raise ProviderAuthError("anthropic")

        async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # pragma: no cover
            return self.chat_sync(messages, **kwargs)

    policy = make_policy(FakeAuthBrokenProvider(), tier="basic")

    class FakeObs:
        prompt_text = "..."

    action = policy(FakeObs())
    assert action == {"action_type": "escalate"}


def test_make_policy_max_tier_uses_max_observation_renderer() -> None:
    fake = _FakeProvider('{"action_type":"escalate"}')
    policy = make_policy(fake, tier="max")

    from dataclasses import dataclass

    @dataclass
    class FakeMaxObs:
        family_id: str = "ecommerce_vibecoded_saas"
        chaos: str = "deploy_regression"
        tick_count: int = 1
        max_ticks: int = 25
        incident_summary: str = "test"
        services: dict[str, dict[str, Any]] = None
        cause_removed: bool = False
        blast_radius: int = 0
        last_log: str = "..."

    obs = FakeMaxObs(services={"api-gateway": {"status": "healthy", "cpu_pct": 30.0, "error_rate_pct": 0.0, "latency_ms": 30.0}})
    policy(obs)
    user_text = fake.last_messages[1]["content"]
    assert "FAMILY:" in user_text or "CHAOS:" in user_text
