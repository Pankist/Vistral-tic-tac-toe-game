"""Template announcer — every agent utterance, deterministic and instant.

Phrase sets come from the game package (one file, editable live). Variants
are picked at random so two consecutive games don't sound canned.
ANNOUNCER=llm may replace the reaction lines (human-mark reaction and
game-over summary) with a generated one-liner under a hard 2s budget —
everything else stays template; the LLM is never on the critical path.
"""

import random

from server.core import config as cfg
from server.core.llm import LLMUnavailable

LLM_POINTS = {"react", "game_over"}   # only these may be flavored


class Announcer:
    def __init__(self, phrases: dict[str, list[str]], mode: str = "template",
                 client=None, log=None):
        self.phrases = phrases
        self.mode = mode
        self.client = client
        self.log = log or (lambda *_: None)
        self._rng = random.Random()

    def render(self, point: str, **ctx) -> str:
        # convenience derivations available to all templates
        if "cell" in ctx:
            ctx.setdefault("cell_cap", str(ctx["cell"]).capitalize())
        if "result" in ctx:
            ctx.setdefault("result_lower", str(ctx["result"]).lower())
        template = self._rng.choice(self.phrases[point])
        text = template.format(**ctx)
        if self.mode == "llm" and point in LLM_POINTS and self.client:
            text = self._flavor(point, text, ctx)
        return text

    def _flavor(self, point: str, fallback: str, ctx) -> str:
        try:
            resp = self.client.chat(
                cfg.MODEL_ANNOUNCER,
                [{"role": "system", "content":
                    "You voice a dry, confident board-game agent. Reply with one "
                    "short declarative line. No exclamation marks, no filler."},
                 {"role": "user", "content":
                    f"Event: {point}. Context: {ctx}. Base line: {fallback!r}. "
                    "Rewrite it with personality, one line."}],
                timeout=cfg.ANNOUNCER_BUDGET_S,
            )
            line = resp["content"].strip().splitlines()[0][:200]
            self.log("announcer_llm", {"model": resp["model"],
                                       "latency_ms": resp["latency_ms"]})
            return line or fallback
        except Exception:
            return fallback   # hard fallback to template, never block
