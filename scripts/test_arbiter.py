"""One-shot arbiter round-trip on a synthetic board photo.

Draws an X and an O on a rendered grid, sends it through the real Arbiter
(needs ANTHROPIC_API_KEY in .env), prints the strict-JSON read and the cost
metadata. Acceptance check #4.

Run: .venv/bin/python scripts/test_arbiter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from server.core.arbiter import Arbiter
from server.core.llm import AnthropicClient
from server.core.loader import load_prompts_module


def synthetic_board() -> np.ndarray:
    img = np.full((330, 330), 235, np.uint8)
    for i in (110, 220):
        cv2.line(img, (i, 8), (i, 322), 40, 5)
        cv2.line(img, (8, i), (322, i), 40, 5)
    # X in cell 0, O in cell 4
    cv2.line(img, (25, 25), (85, 85), 20, 7)
    cv2.line(img, (85, 25), (25, 85), 20, 7)
    cv2.circle(img, (165, 165), 32, 20, 7)
    return img


def main() -> None:
    calls = []
    arbiter = Arbiter(AnthropicClient(), load_prompts_module(),
                      log=lambda kind, data: calls.append((kind, data)))
    board = ["X", "", "", "", "", "", "", "", ""]
    result = arbiter.read_board(synthetic_board(), board,
                                changed="cell 4 read as O with low confidence")
    print("cells:")
    for c in result["cells"]:
        print(f"  {c['cell']}: {c['mark'] or 'empty':5s} conf {c['conf']}")
    print(f"rationale: {result.get('rationale', '')}")
    print(f"model: {result['model']}")
    for kind, data in calls:
        print(f"logged {kind}: latency {data.get('latency_ms')}ms "
              f"usage {data.get('usage')}")
    expected = {0: "X", 4: "O"}
    got = {c["cell"]: c["mark"] for c in result["cells"]}
    ok = all(got.get(k) == v for k, v in expected.items()) and \
        all(got[i] == "" for i in range(9) if i not in expected)
    print("PASS" if ok else "MISMATCH — inspect above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
