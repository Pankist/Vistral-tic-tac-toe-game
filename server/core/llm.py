"""OpenRouter client — the only piece of the system that talks to a model.

One method, strict-JSON oriented, hard timeout, cost logged by the caller.
No key -> LLMUnavailable, and every caller has a non-LLM fallback path.
"""

import json
import time

import httpx

from server.core import config as cfg

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMUnavailable(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout: float | None = None):
        self.api_key = api_key if api_key is not None else cfg.OPENROUTER_API_KEY
        self.timeout = timeout or cfg.LLM_TIMEOUT_S

    def chat(self, model: str, messages: list[dict], timeout: float | None = None) -> dict:
        """Returns {"content": str, "model": str, "latency_ms": int, "usage": {...}}.

        `model` in the result is the resolved id as returned by the API — part
        of the provenance tuple, not just the requested string.
        """
        if not self.api_key:
            raise LLMUnavailable("OPENROUTER_API_KEY not set")
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": messages, "usage": {"include": True}},
                timeout=timeout or self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise LLMUnavailable(str(e)) from e
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMUnavailable(f"malformed response: {e}") from e
        return {
            "content": content,
            "model": data.get("model", model),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "usage": data.get("usage", {}),
        }


def parse_strict_json(text: str) -> dict:
    """Tolerate accidental code fences, nothing else."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)
