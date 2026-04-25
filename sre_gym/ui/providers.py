"""Provider abstractions for BYOK chat-completion routing.

Every provider implements a single async method, ``chat(messages, **kwargs)``,
returning the assistant text. Errors raise typed exceptions from
``sre_gym.exceptions`` so the UI can surface a redacted message rather than
echoing the failing key.

Three implementations:

- ``HFInferenceProvider``     — huggingface_hub.InferenceClient
- ``AnthropicProvider``       — official anthropic SDK
- ``OpenAICompatibleProvider``— OpenAI/Together/Fireworks/Groq/DeepSeek

All three are sync-callable too via ``provider.chat_sync(...)``; the Gradio
streaming path uses the sync surface to keep the per-tick yield loop simple.

Security
--------
Tokens passed to ``__init__`` live only on the instance. They are never
logged, never written to disk, and never returned in error messages.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from sre_gym.exceptions import (
    ProviderAuthError,
    ProviderModelError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)


class Provider(Protocol):
    """All providers expose this surface."""

    name: str
    model: str

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # pragma: no cover - protocol
        ...

    def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Hugging Face Inference Client.
# ---------------------------------------------------------------------------


class HFInferenceProvider:
    """Routes chat completions through huggingface_hub.InferenceClient.

    Works with any HF-hosted model that supports the chat-completion task,
    including the Inference Router (which fans out to providers like Together,
    Fireworks, Novita, Replicate based on model availability).
    """

    name = "hf"

    def __init__(self, hf_token: str, model: str) -> None:
        if not hf_token:
            raise ProviderAuthError(self.name, "HF token required")
        self._token = hf_token
        self.model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ProviderModelError(self.name, f"huggingface_hub not installed: {exc}") from exc
        self._client = InferenceClient(model=self.model, token=self._token)
        return self._client

    def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        client = self._ensure_client()
        try:
            resp = client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.0),
            )
        except Exception as exc:
            return _classify_provider_exception(self.name, exc)
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover
            raise ProviderModelError(self.name, f"unexpected response shape: {exc}") from exc

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        # InferenceClient is sync; offload to a worker thread.
        import asyncio
        return await asyncio.to_thread(self.chat_sync, messages, **kwargs)


# ---------------------------------------------------------------------------
# Anthropic SDK.
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Routes chat completions through the official Anthropic SDK.

    Default model is Claude Sonnet 4.6 (current at submission time, 2026-04).
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        if not api_key:
            raise ProviderAuthError(self.name, "Anthropic API key required")
        self._api_key = api_key
        self.model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderModelError(self.name, "anthropic SDK not installed") from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        client = self._ensure_client()
        # Split off any system message — Anthropic uses a top-level system param.
        system_text: str | None = None
        thread: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system" and system_text is None:
                system_text = msg.get("content", "")
            else:
                thread.append({"role": msg["role"], "content": msg.get("content", "")})
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 256),
                system=system_text or "",
                messages=thread,
            )
        except Exception as exc:
            return _classify_provider_exception(self.name, exc)
        try:
            return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
        except Exception as exc:  # pragma: no cover
            raise ProviderModelError(self.name, f"unexpected response shape: {exc}") from exc

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import asyncio
        return await asyncio.to_thread(self.chat_sync, messages, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI-compatible (covers OpenAI, Groq, Together, Fireworks, DeepSeek).
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """Routes chat completions through any OpenAI-compatible endpoint.

    The OpenAI Python SDK speaks the chat-completion protocol that Together,
    Groq, Fireworks, DeepSeek, and OpenAI all expose. ``base_url`` is the
    knob that picks the provider.

    Examples
    --------
    >>> OpenAICompatibleProvider("https://api.openai.com/v1", api_key="...", model="gpt-5")
    >>> OpenAICompatibleProvider("https://api.together.xyz/v1", api_key="...",
    ...                          model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    >>> OpenAICompatibleProvider("https://api.groq.com/openai/v1", api_key="...",
    ...                          model="llama-3.3-70b-versatile")
    >>> OpenAICompatibleProvider("https://api.fireworks.ai/inference/v1", api_key="...",
    ...                          model="accounts/fireworks/models/llama-v3p3-70b-instruct")
    >>> OpenAICompatibleProvider("https://api.deepseek.com/v1", api_key="...",
    ...                          model="deepseek-chat")
    """

    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        if not api_key:
            raise ProviderAuthError(self.name, "API key required")
        if not base_url:
            raise ProviderModelError(self.name, "base_url required")
        self._base_url = base_url
        self._api_key = api_key
        self.model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderModelError(self.name, "openai SDK not installed") from exc
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def chat_sync(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        client = self._ensure_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.0),
            )
        except Exception as exc:
            return _classify_provider_exception(self.name, exc)
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover
            raise ProviderModelError(self.name, f"unexpected response shape: {exc}") from exc

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import asyncio
        return await asyncio.to_thread(self.chat_sync, messages, **kwargs)


# ---------------------------------------------------------------------------
# Exception classifier.
# ---------------------------------------------------------------------------


def _classify_provider_exception(provider_name: str, exc: BaseException) -> str:
    """Map any provider exception to one of our typed errors and re-raise.

    Returns a never-actually-returned ``str`` to keep type-checkers happy when
    used inline in ``return`` paths.
    """
    msg = str(exc).lower()
    # Auth-class errors
    auth_markers = ("401", "unauthorized", "auth", "invalid api key", "incorrect api key", "forbidden", "403")
    if any(m in msg for m in auth_markers):
        raise ProviderAuthError(provider_name, "auth failed") from exc
    # Rate-limit errors
    rl_markers = ("429", "rate limit", "too many requests", "quota")
    if any(m in msg for m in rl_markers):
        raise ProviderRateLimitError(provider_name) from exc
    # Generic — strip any tokens that might appear in the message before re-raising.
    safe = str(exc).split("Bearer ")[0]
    raise ProviderModelError(provider_name, safe[:200]) from exc
