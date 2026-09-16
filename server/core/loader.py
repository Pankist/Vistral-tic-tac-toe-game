"""Resolves ACTIVE_GAME and the game's DEFAULT_ENGINE to concrete objects.
The only place that maps names to classes — everything else works through
the protocols.
"""

import importlib

from server.core import config as cfg
from server.core.engine_llm import LLMEngine
from server.core.llm import OpenRouterClient
from server.core.protocols import Engine, Game


def load_game() -> Game:
    mod = importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.game")
    # convention: the game module exposes exactly one Game class, named after it
    cls = getattr(mod, "TicTacToe") if cfg.ACTIVE_GAME == "tic_tac_toe" else None
    if cls is None:
        raise RuntimeError(f"no game class registered for {cfg.ACTIVE_GAME}")
    return cls()


def load_marks_module():
    return importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.marks")


def load_prompts_module():
    return importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.prompts")


def load_phrases() -> dict:
    return importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.phrases").PHRASES


def load_engine(game: Game, client: OpenRouterClient | None = None, log=None) -> Engine:
    gcfg = importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.config")
    if cfg.ACTIVE_GAME == "tic_tac_toe":
        from server.games.tic_tac_toe.engine_minimax import MinimaxEngine
        native = MinimaxEngine(mark=game.agent_mark, strength=gcfg.STRENGTH)
    else:
        raise RuntimeError(f"no native engine for {cfg.ACTIVE_GAME}")
    if gcfg.DEFAULT_ENGINE == "llm":
        return LLMEngine(game, client or OpenRouterClient(),
                         fallback=native, prompts=load_prompts_module(), log=log)
    return native
