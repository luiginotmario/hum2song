"""One OpenAI-compatible OpenRouter client (SPEC §8)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hum2song.decide.config import DecideConfig


class OpenRouterError(RuntimeError):
    """OpenRouter call failed after retries."""


def chat(
    config: DecideConfig,
    messages: list[dict[str, str]],
    model: str | None = None,
    *,
    json_mode: bool = False,
) -> str:
    """POST /chat/completions. Raises OpenRouterError when the key is missing or the call fails."""
    if not config.openrouter_ready:
        raise OpenRouterError("OPENROUTER_API_KEY is not set")
    body: dict[str, Any] = {
        "model": model or config.llm_model,
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None
    for _attempt in range(2):
        request = urllib.request.Request(
            f"{config.openrouter_base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/luiginotmario/hum2song",
                "X-Title": "hum2song",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.openrouter_timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
    raise OpenRouterError(str(last_error))
