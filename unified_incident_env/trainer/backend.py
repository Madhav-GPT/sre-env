"""Backend interfaces for model calls."""

from __future__ import annotations

import json
import time
from typing import Protocol
from urllib.parse import urlparse

from openai import OpenAI

from .types import ModelRequest, ModelResponse


class ModelBackend(Protocol):
    """Minimal backend protocol for trainer use."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return raw model text and metadata for one request."""


class OpenAICompatibleBackend:
    """OpenAI-compatible backend, suitable for Ollama and similar servers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 90.0,
    ) -> None:
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def complete(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        create_kwargs = {
            "model": request.model_name,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        raw_text = ""
        actual_mode = request.structured_mode

        if request.structured_mode == "backend_adaptive":
            if self._is_ollama():
                actual_mode = "response_format_json"
            else:
                actual_mode = "tool_calling"

        try:
            if actual_mode == "tool_calling":
                tool_choice = request.tool_choice or {
                    "type": "function",
                    "function": {"name": "emit_action"},
                }
                create_kwargs["tools"] = request.tools or [
                    self._tool_from_response_format(request.response_format)
                ]
                create_kwargs["tool_choice"] = tool_choice
                response = self._client.chat.completions.create(**create_kwargs)
                raw_text = self._extract_tool_text(response)
            elif actual_mode == "response_format_json":
                if self._is_ollama():
                    create_kwargs["extra_body"] = {
                        "format": self._ollama_format_payload(request.response_format)
                    }
                else:
                    create_kwargs["response_format"] = request.response_format or {
                        "type": "json_object"
                    }
                response = self._client.chat.completions.create(**create_kwargs)
                raw_text = response.choices[0].message.content or ""
            else:
                response = self._client.chat.completions.create(**create_kwargs)
                raw_text = response.choices[0].message.content or ""
        except Exception:
            if request.structured_mode == "backend_adaptive" and actual_mode == "tool_calling":
                fallback_kwargs = dict(create_kwargs)
                if "tools" in fallback_kwargs:
                    del fallback_kwargs["tools"]
                if "tool_choice" in fallback_kwargs:
                    del fallback_kwargs["tool_choice"]
                if self._is_ollama():
                    fallback_kwargs["extra_body"] = {
                        "format": self._ollama_format_payload(request.response_format)
                    }
                else:
                    fallback_kwargs["response_format"] = request.response_format or {
                        "type": "json_object"
                    }
                response = self._client.chat.completions.create(**fallback_kwargs)
                raw_text = response.choices[0].message.content or ""
                actual_mode = "response_format_json"
            else:
                raise

        elapsed = time.perf_counter() - started
        return ModelResponse(
            raw_text=raw_text,
            latency_s=round(elapsed, 4),
            metadata={
                "model": request.model_name,
                "structured_mode": actual_mode,
            },
        )

    def _is_ollama(self) -> bool:
        parsed = urlparse(self.base_url)
        host = parsed.netloc.lower()
        return "11434" in host or "ollama" in host or "127.0.0.1" in host or "localhost" in host

    def _ollama_format_payload(
        self,
        response_format: dict[str, object] | None,
    ) -> object:
        if response_format and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema", {})
            if isinstance(json_schema, dict):
                return json_schema.get("schema", "json")
        return "json"

    def _tool_from_response_format(
        self,
        response_format: dict[str, object] | None,
    ) -> dict[str, object]:
        schema = {
            "type": "object",
            "properties": {"action_type": {"type": "string"}},
            "required": ["action_type"],
            "additionalProperties": False,
        }
        if response_format and response_format.get("type") == "json_schema":
            json_schema = response_format.get("json_schema", {})
            schema = json_schema.get("schema", schema)  # type: ignore[assignment]
        return {
            "type": "function",
            "function": {
                "name": "emit_action",
                "description": "Emit exactly one structured environment action.",
                "parameters": schema,
            },
        }

    def _extract_tool_text(self, response) -> str:
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            function = getattr(tool_calls[0], "function", None)
            if function is not None and getattr(function, "arguments", None):
                return function.arguments
        content = message.content or ""
        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    fragments.append(item.get("text", ""))
            return "".join(fragments)
        if isinstance(content, str):
            return content
        return json.dumps(content)
