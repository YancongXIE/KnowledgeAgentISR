from __future__ import annotations
import json
import os
import time
from typing import Any, Type, TypeVar
from pydantic import BaseModel
T = TypeVar('T', bound=BaseModel)

def _trace_enabled() -> bool:
    return os.environ.get('KG_TRACE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}

def _trace(msg: str) -> None:
    if _trace_enabled():
        print(f'[llm] {msg}', flush=True)

def _is_retryable_server_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return '500' in msg or '503' in msg or 'UNAVAILABLE' in msg or 'HIGH DEMAND' in msg or 'TIMEOUT' in msg or 'CONNECTION RESET' in msg or '429' in msg or 'RATE' in msg

def complete_json(client: Any, *, model: str, system: str, user: str, schema_model: Type[T], temperature: float = 0.0) -> T:
    """Call the LLM client to produce a validated Pydantic object from JSON output.

    The `client` must implement a `.chat(...)` method that returns
    {"message": {"content": "<json string>"}}.
    """
    guide = (
        f'Respond with a single JSON object only, no markdown fences. '
        f'Required top-level keys: {list(schema_model.model_json_schema().get("properties", {}).keys())}.'
    )
    max_retries = 3
    base_sleep_s = 2.0
    max_tokens = int(os.environ.get('KG_JSON_MAX_TOKENS', '1024'))
    raw = None
    messages = [
        {'role': 'system', 'content': system + '\n\n' + guide},
        {'role': 'user', 'content': user},
    ]
    for attempt in range(max_retries):
        try:
            _trace(f'request model={model} schema={schema_model.__name__} attempt={attempt + 1}/{max_retries}')
            resp = client.chat(
                model=model,
                messages=messages,
                format='json',
                options={'temperature': temperature, 'num_predict': max_tokens},
            )
            raw = ((resp.get('message', {}) if isinstance(resp, dict) else {}).get('content', '') or '{}').strip()
            _trace(f'response model={model} schema={schema_model.__name__} chars={len(raw)}')
            try:
                data = json.loads(raw)
            except Exception as exc:
                is_last = attempt == max_retries - 1
                if is_last:
                    _trace(f'error model={model} final=True invalid_json={exc}')
                    raise
                messages.append({'role': 'assistant', 'content': raw})
                messages.append({'role': 'user', 'content': 'Your previous response was not valid JSON. Return exactly one JSON object with all required keys and no extra text.'})
                continue
            try:
                return schema_model.model_validate(data)
            except Exception as exc:
                is_last = attempt == max_retries - 1
                if is_last:
                    _trace(f'error model={model} final=True schema_validation={exc}')
                    raise
                messages.append({'role': 'assistant', 'content': raw})
                messages.append({'role': 'user', 'content': f'Your JSON did not satisfy the required schema. Return exactly one JSON object that includes all required keys: {list(schema_model.model_json_schema().get("properties", {}).keys())}. Do not omit required fields.'})
                continue
        except Exception as exc:
            is_last = attempt == max_retries - 1
            if not _is_retryable_server_error(exc) or is_last:
                _trace(f'error model={model} final={is_last} err={exc}')
                raise
            sleep_s = base_sleep_s * 2 ** attempt
            _trace(f'retrying in {sleep_s:.1f}s after err={exc}')
            time.sleep(sleep_s)
    if raw is None:
        raise RuntimeError('LLM call did not return a response.')
    data = json.loads(raw)
    return schema_model.model_validate(data)
