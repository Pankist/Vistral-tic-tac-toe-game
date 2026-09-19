"""Resolves ACTIVE_GAME and the game's DEFAULT_ENGINE to concrete objects.
The only place that maps names to classes — everything else works through
the protocols.
"""

import importlib

from server.core import config as cfg
from server.core.engine_llm import LLMEngine
from server.core.llm import AnthropicClient
from server.core.protocols import Engine, Game


def load_game(game_name: str = None) -> Game:
    game = game_name or cfg.ACTIVE_GAME
    mod = importlib.import_module(f"server.games.{game}.game")
    # convention: game module exposes one Game class
    if game == "tic_tac_toe":
        cls = getattr(mod, "TicTacToe")
    elif game == "submarine":
        cls = getattr(mod, "Submarine")
    else:
        raise RuntimeError(f"no game class registered for {game}")
    return cls()


def load_marks_module(game_name: str = None):
    game = game_name or cfg.ACTIVE_GAME
    try:
        return importlib.import_module(f"server.games.{game}.marks")
    except ImportError:
        return None  # Submarine doesn't have marks


def load_prompts_module(game_name: str = None):
    game = game_name or cfg.ACTIVE_GAME
    return importlib.import_module(f"server.games.{game}.prompts")


def load_phrases(game_name: str = None) -> dict:
    game = game_name or cfg.ACTIVE_GAME
    return importlib.import_module(f"server.games.{game}.phrases").PHRASES


def load_engine(game: Game, client: AnthropicClient | None = None, log=None, game_name: str = None) -> Engine:
    gname = game_name or game.name
    gcfg = importlib.import_module(f"server.games.{gname}.config")
    if gname == "tic_tac_toe":
        from server.games.tic_tac_toe.engine_minimax import MinimaxEngine
        native = MinimaxEngine(mark=game.agent_mark, strength=gcfg.STRENGTH)
    elif gname == "submarine":
        from server.games.submarine.engine import SubmarineEngine
        return SubmarineEngine(client, log=log)
    else:
        raise RuntimeError(f"no native engine for {gname}")
    if getattr(gcfg, 'DEFAULT_ENGINE', None) == "llm":
        return LLMEngine(game, client or AnthropicClient(),
                         fallback=native, prompts=load_prompts_module(gname), log=log)
    return native
