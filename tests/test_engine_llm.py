"""Generic LLM engine with a mocked client — incl. the illegal-move fallback."""

import json

from server.core.engine_llm import LLMEngine
from server.core.llm import LLMUnavailable
from server.core.loader import load_prompts_module
from server.games.tic_tac_toe.engine_minimax import MinimaxEngine
from server.games.tic_tac_toe.game import TicTacToe

game = TicTacToe()
prompts = load_prompts_module()


class MockClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, messages, timeout=None):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"content": r, "model": model + ":resolved",
                "latency_ms": 5, "usage": {"cost": 0.001}}


def make(replies, log=None):
    return LLMEngine(game, MockClient(replies),
                     fallback=MinimaxEngine(mark="O"), prompts=prompts, log=log)


def test_legal_move_accepted_with_reasoning():
    e = make([json.dumps({"cell": 4, "reasoning": "take the center"})])
    b = ["X"] + [""] * 8
    assert e.decide(b, []) == 4
    assert e.last_reasoning == "take the center"


def test_illegal_move_retried_then_accepted():
    e = make([json.dumps({"cell": 0, "reasoning": "occupied, oops"}),
              json.dumps({"cell": 4, "reasoning": "fine, center"})])
    b = ["X"] + [""] * 8
    assert e.decide(b, []) == 4


def test_illegal_twice_falls_back_to_minimax():
    logged = []
    e = make([json.dumps({"cell": 0, "reasoning": "bad"}),
              json.dumps({"cell": 99, "reasoning": "worse"})],
             log=lambda kind, data: logged.append(kind))
    b = ["X"] + [""] * 8
    move = e.decide(b, [])
    assert move in game.legal_moves(b)
    assert move == MinimaxEngine(mark="O").decide(b, [])
    assert "engine_llm_fallback" in logged


def test_malformed_json_falls_back():
    e = make(["the center looks nice", "still not json"])
    b = ["X"] + [""] * 8
    assert e.decide(b, []) in game.legal_moves(b)


def test_llm_unavailable_falls_back():
    e = make([LLMUnavailable("no key"), LLMUnavailable("no key")])
    b = ["X"] + [""] * 8
    assert e.decide(b, []) in game.legal_moves(b)


def test_code_fenced_json_tolerated():
    e = make(['```json\n{"cell": 4, "reasoning": "center"}\n```'])
    assert e.decide(["X"] + [""] * 8, []) == 4
