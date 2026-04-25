"""Smoke tests for the four providers (UI Build Addendum §3.4).

Confirms each provider:

1. Constructs without network calls.
2. Maps the right ``base_url`` for OpenAI-compatible providers (Together /
   Fireworks / Groq / DeepSeek / OpenAI).
3. Refuses empty credentials with ``ProviderAuthError`` (security hardening).
4. ``DummyProvider`` returns a valid JSON action without any network call.
"""

from __future__ import annotations

import json

import pytest

from sre_gym.exceptions import ProviderAuthError, ProviderModelError
from sre_gym.ui.providers import (
    AnthropicProvider,
    DummyProvider,
    HFInferenceProvider,
    OpenAICompatibleProvider,
)
from sre_gym.ui.router import (
    ProviderKind,
    build_provider,
    find_entry,
    models_for_tier,
)
from sre_gym.tier import Tier


# ---------- Construction + base_url mapping ----------


def test_hf_provider_constructs_with_token_and_model() -> None:
    p = HFInferenceProvider(hf_token="hf_test", model="Qwen/Qwen2.5-7B-Instruct")
    assert p.name == "hf"
    assert p.model == "Qwen/Qwen2.5-7B-Instruct"


def test_anthropic_provider_constructs_with_key() -> None:
    p = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-6")
    assert p.name == "anthropic"
    assert p.model == "claude-sonnet-4-6"


def test_openai_compat_provider_uses_addendum_base_urls() -> None:
    """Per addendum §3.4 the base_urls for each OpenAI-compatible vendor."""
    cases = [
        ("https://api.openai.com/v1", "gpt-5"),
        ("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
        ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("https://api.deepseek.com/v1", "deepseek-chat"),
    ]
    for base, model in cases:
        p = OpenAICompatibleProvider(base_url=base, api_key="sk-test", model=model)
        assert p.name == "openai-compat"
        assert p._base_url == base
        assert p.model == model


def test_dummy_provider_returns_valid_action_json_synchronously() -> None:
    """DummyProvider must produce a parseable JSON action with no network."""
    p = DummyProvider()
    raw = p.chat_sync(messages=[{"role": "user", "content": "anything"}])
    obj = json.loads(raw)
    assert "action_type" in obj


def test_dummy_provider_cycles_through_default_actions() -> None:
    """The default cycle is the scripted-optimal trajectory for a Basic episode."""
    p = DummyProvider()
    types = []
    for _ in range(len(DummyProvider.DEFAULT_CYCLE)):
        types.append(json.loads(p.chat_sync([{"role": "user", "content": "x"}]))["action_type"])
    assert types[-1] == "declare_resolved"
    assert "rollback_deploy" in types
    assert "submit_hypothesis" in types


def test_dummy_provider_accepts_custom_responses() -> None:
    custom = (
        {"action_type": "escalate"},
        {"action_type": "query_logs", "service": "cache"},
    )
    p = DummyProvider(responses=custom)
    assert json.loads(p.chat_sync([]))["action_type"] == "escalate"
    assert json.loads(p.chat_sync([]))["action_type"] == "query_logs"


@pytest.mark.asyncio
async def test_dummy_provider_async_path_returns_same_payload() -> None:
    p = DummyProvider()
    text = await p.chat([{"role": "user", "content": "x"}])
    obj = json.loads(text)
    assert "action_type" in obj


# ---------- Auth refusal (security hardening) ----------


def test_hf_provider_refuses_empty_token() -> None:
    with pytest.raises(ProviderAuthError):
        HFInferenceProvider(hf_token="", model="Qwen/Qwen2.5-7B-Instruct")


def test_anthropic_provider_refuses_empty_key() -> None:
    with pytest.raises(ProviderAuthError):
        AnthropicProvider(api_key="", model="claude-sonnet-4-6")


def test_openai_compat_provider_refuses_empty_key() -> None:
    with pytest.raises(ProviderAuthError):
        OpenAICompatibleProvider(base_url="https://api.openai.com/v1", api_key="", model="gpt-5")


def test_openai_compat_provider_refuses_empty_base_url() -> None:
    with pytest.raises(ProviderModelError):
        OpenAICompatibleProvider(base_url="", api_key="sk-test", model="gpt-5")


# ---------- Router-driven provider construction ----------


def test_router_build_provider_basic_default_returns_hf_provider() -> None:
    entry = models_for_tier(Tier.BASIC)[0]
    assert entry.kind is ProviderKind.HF
    p = build_provider(entry, hf_token="hf_test")
    assert isinstance(p, HFInferenceProvider)


def test_router_build_provider_max_default_returns_anthropic_provider() -> None:
    entry = models_for_tier(Tier.MAX)[0]
    assert entry.kind is ProviderKind.ANTHROPIC
    p = build_provider(entry, anthropic_key="sk-ant-test")
    assert isinstance(p, AnthropicProvider)


def test_router_build_provider_advanced_default_returns_openai_compat() -> None:
    """Advanced tier's default is Llama-3.3-70B Together (BYOK)."""
    entry = models_for_tier(Tier.ADVANCED)[0]
    assert entry.kind is ProviderKind.OPENAI_COMPAT
    assert entry.base_url == "https://api.together.xyz/v1"
    p = build_provider(entry, together_key="sk-test")
    assert isinstance(p, OpenAICompatibleProvider)
    assert p._base_url == "https://api.together.xyz/v1"


def test_router_find_entry_resolves_label_or_model_id_for_each_tier() -> None:
    for tier in Tier:
        for entry in models_for_tier(tier):
            assert find_entry(entry.label, tier) is entry
            assert find_entry(entry.model_id, tier) is entry
