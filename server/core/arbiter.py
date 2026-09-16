"""Low-confidence board read via a vision model — the one expensive call in
the loop, and it runs only when classical CV admits uncertainty.

Input: rectified board JPEG + current confirmed board + what changed.
Output: 9 cells with per-cell confidence + one-line rationale, strict JSON.
Timeout 6s, one retry, then the FSM falls back to asking the human.
"""

import base64

import cv2

from server.core import config as cfg
from server.core.llm import LLMUnavailable, OpenRouterClient, parse_strict_json


class ArbiterUnavailable(Exception):
    pass


class Arbiter:
    def __init__(self, client: OpenRouterClient, prompts, log=None):
        self.client = client
        self.prompts = prompts
        self.log = log or (lambda *_: None)

    def read_board(self, rectified, board: list[str], changed: str) -> dict:
        """rectified: grayscale canonical board image (numpy). Returns
        {"cells": [{"cell","mark","conf"}...], "rationale": str, "model": str}.
        """
        ok, buf = cv2.imencode(".jpg", rectified, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ArbiterUnavailable("could not encode rectified frame")
        b64 = base64.b64encode(buf.tobytes()).decode()
        messages = [
            {"role": "system", "content": self.prompts.ARBITER_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": self.prompts.ARBITER_USER.format(
                    board=board, changed=changed)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ]
        last_err = None
        for attempt in range(2):
            try:
                resp = self.client.chat(cfg.MODEL_ARBITER, messages)
                parsed = parse_strict_json(resp["content"])
                cells = parsed["cells"]
                assert len(cells) == 9
                for c in cells:
                    assert c["mark"] in ("X", "O", "")
                    assert 0.0 <= float(c["conf"]) <= 1.0
                self.log("arbiter_call", {
                    "model": resp["model"], "latency_ms": resp["latency_ms"],
                    "usage": resp["usage"], "attempt": attempt,
                    "rationale": parsed.get("rationale", ""),
                })
                parsed["model"] = resp["model"]
                return parsed
            except (LLMUnavailable, ValueError, KeyError, AssertionError) as e:
                last_err = e
        raise ArbiterUnavailable(str(last_err))
