"""Anthropic API client — the only piece of the system that talks to a model.

One method, strict-JSON oriented, hard timeout, cost logged by the caller.
No key -> LLMUnavailable, and every caller has a non-LLM fallback path.

Callers build provider-neutral (OpenAI-style) message lists — system/user
roles, text parts, data-URL image parts — and this client translates them to
the Messages API. That keeps prompts and call sites provider-agnostic; a
provider change is this one file.
"""

import json
import time

import httpx

from server.core import config as cfg

API_URL = "https://api.anthropic.com/v1/messages"


class LLMUnavailable(Exception):
    pass


class AnthropicClient:
    def __init__(self, api_key: str | None = None, timeout: float | None = None):
        self.api_key = api_key if api_key is not None else cfg.ANTHROPIC_API_KEY
        self.timeout = timeout or cfg.LLM_TIMEOUT_S

    def chat(self, model: str, messages: list[dict], timeout: float | None = None) -> dict:
        """Returns {"content": str, "model": str, "latency_ms": int, "usage": {...}}.

        `model` in the result is the resolved id as returned by the API — part
        of the provenance tuple, not just the requested string.
        """
        if not self.api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")
        system, converted = _convert(messages)
        payload = {"model": model, "max_tokens": 1024, "messages": converted}
        if system:
            payload["system"] = system
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                API_URL,
                headers={"x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01"},
                json=payload,
                timeout=timeout or self.timeout,
            )
            print(f"[LLM] Response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            print(f"[LLM] Response data keys: {data.keys()}")
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            print(f"[LLM ERROR] HTTP/JSON error: {e}")
            raise LLMUnavailable(str(e)) from e
        try:
            content = "".join(b["text"] for b in data["content"]
                              if b.get("type") == "text")
            print(f"[LLM] Extracted content length: {len(content)}")
            if not content:
                print(f"[LLM WARNING] Empty content! data['content']: {data.get('content')}")
        except (KeyError, TypeError) as e:
            print(f"[LLM ERROR] Content extraction failed: {e}")
            raise LLMUnavailable(f"malformed response: {e}") from e
        return {
            "content": content,
            "model": data.get("model", model),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "usage": data.get("usage", {}),
        }


def _convert(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-style messages -> (system string, Anthropic message list)."""
    system_parts, out = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
            continue
        content = m["content"]
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = []
            for part in content:
                if part["type"] == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]     # data:image/jpeg;base64,...
                    header, _, b64 = url.partition(";base64,")
                    blocks.append({"type": "image", "source": {
                        "type": "base64",
                        "media_type": header.removeprefix("data:"),
                        "data": b64}})
        out.append({"role": m["role"], "content": blocks})
    return "\n".join(system_parts), out


def parse_strict_json(text: str) -> dict:
    """Tolerate accidental code fences, nothing else."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)
