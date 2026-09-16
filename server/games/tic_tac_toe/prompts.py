"""LLM prompts owned by this game: the arbiter vision prompt and the rules
text used by the generic LLM engine. Strict JSON contracts — the callers
validate and fall back on anything malformed.
"""

ARBITER_SYSTEM = (
    "You read a rectified photo of a hand-drawn tic-tac-toe board. "
    "Respond with strict JSON only, no prose, no markdown fences."
)

ARBITER_USER = """This is a 330x330 rectified photo of a 3x3 tic-tac-toe grid drawn on paper.
Cells are numbered 0-8 row-major: 0 top-left, 4 center, 8 bottom-right.

The last confirmed board state was:
{board}

What changed since then, according to my (possibly wrong) classical CV read: {changed}

Read all 9 cells from the image. Reply with strict JSON exactly in this shape:
{{"cells": [{{"cell": 0, "mark": "X"|"O"|"", "conf": 0.0}}, ... 9 entries ...], "rationale": "one line"}}"""


def engine_system(rules_text: str) -> str:
    return (
        "You are playing a board game against a human. Rules:\n"
        f"{rules_text}\n"
        "You must reply with strict JSON only: {\"cell\": <int>, \"reasoning\": \"one short line\"}. "
        "No prose outside the JSON, no markdown fences."
    )


def engine_user(serialized_board: str, mark: str, legal: list[int]) -> str:
    return (
        f"Board now ('.' = empty):\n{serialized_board}\n"
        f"You play '{mark}'. Legal cells: {legal}.\n"
        "Pick one legal cell and reply with the JSON."
    )
