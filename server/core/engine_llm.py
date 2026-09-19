"""Generic LLM engine — game-agnostic by construction.

It gets rules text, board serialization, and legal moves from the Game
object; it never imports a concrete game. One chat call, strict JSON
{cell, reasoning} out; the reasoning line feeds the announcer. On illegal or
malformed output: one retry carrying the validation error, then fall back to
the game's default engine for that turn and log it. The FSM validates every
move regardless of engine — a hallucinating model cannot corrupt state.
"""

from server.core import config as cfg
from server.core.llm import LLMUnavailable, AnthropicClient, parse_strict_json
from server.core.protocols import Board, Engine, Game, Move


class LLMEngine:
    name = "llm"

    def __init__(self, game: Game, client: AnthropicClient,
                 fallback: Engine, prompts, log=None):
        self.game = game
        self.client = client
        self.fallback = fallback
        self.prompts = prompts
        self.log = log or (lambda *_: None)
        self.last_reasoning = ""

    def decide(self, board: Board, history: list[tuple[str, Move]]) -> Move:
        legal = self.game.legal_moves(board)
        messages = [
            {"role": "system",
             "content": self.prompts.engine_system(self.game.rules_text())},
            {"role": "user",
             "content": self.prompts.engine_user(
                 self.game.serialize(board), self.game.agent_mark, legal)},
        ]
        error = None
        for attempt in range(2):
            if error:
                messages.append({"role": "user", "content":
                                 f"Your previous reply was invalid: {error}. "
                                 "Reply again with strict JSON, one legal cell."})
            try:
                resp = self.client.chat(cfg.MODEL_ENGINE, messages)
                parsed = parse_strict_json(resp["content"])
                move = int(parsed["cell"])
                if move not in legal:
                    raise ValueError(f"cell {move} is not a legal move")
                self.last_reasoning = str(parsed.get("reasoning", ""))[:200]
                self.log("engine_llm_call", {
                    "model": resp["model"], "latency_ms": resp["latency_ms"],
                    "usage": resp["usage"], "move": move, "attempt": attempt,
                    "reasoning": self.last_reasoning,
                })
                return move
            except (LLMUnavailable, ValueError, KeyError) as e:
                error = str(e)
                messages.append({"role": "assistant", "content": "invalid"})
        move = self.fallback.decide(board, history)
        self.last_reasoning = ""
        self.log("engine_llm_fallback", {"error": error, "fallback_move": move,
                                         "fallback_engine": self.fallback.name})
        return move
