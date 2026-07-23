"""Shared helpers for KDE agent tests."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel


class MockLLMClient:
    """Minimal AzureOpenAIClient stand-in for complete_json()."""

    def __init__(self, responses: Dict[str, dict] | None = None, *, fail: bool = False) -> None:
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "format": format})
        if self.fail:
            raise RuntimeError("mock LLM failure")
        # Match by schema hint in system message (complete_json appends required keys).
        system = messages[0]["content"] if messages else ""
        for key, payload in self.responses.items():
            if key in system or key in json.dumps(messages):
                return {"message": {"content": json.dumps(payload)}}
        # Default: return first registered response or empty object.
        if self.responses:
            first = next(iter(self.responses.values()))
            return {"message": {"content": json.dumps(first)}}
        return {"message": {"content": "{}"}}

    def generate(self, *, model: str, system: str, prompt: str, stream: bool = False) -> dict[str, Any]:
        return {"response": "mock generated text"}


def mock_payload_for(model_cls: Type[BaseModel], **overrides: Any) -> dict[str, Any]:
    base = {}
    for name, field in model_cls.model_fields.items():
        ann = str(field.annotation)
        if "List" in ann or "list" in ann:
            base[name] = []
        elif "bool" in ann:
            base[name] = False
        elif "float" in ann:
            base[name] = 0.5
        else:
            base[name] = ""
    base.update(overrides)
    return base
