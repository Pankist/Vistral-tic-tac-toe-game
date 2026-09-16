"""Memoized full-depth minimax — perfect play, <1ms, $0.

STRENGTH dial: "perfect" plays optimally; "random" plays a uniform legal move.
The dial exists because opponent strength is a named decision in the
assignment; perfect is the documented default.
"""

import random
from functools import lru_cache

from server.games.tic_tac_toe.game import TicTacToe

_game = TicTacToe()


@lru_cache(maxsize=None)
def _score(board: tuple[str, ...], to_move: str, me: str) -> int:
    """+1 win for `me`, -1 loss, 0 draw; depth-free (perfect play either way)."""
    w = _game.winner(list(board))
    if w is not None:
        return 1 if w == me else -1
    if all(c != "" for c in board):
        return 0
    other = "O" if to_move == "X" else "X"
    scores = []
    for m in _game.legal_moves(list(board)):
        child = list(board)
        child[m] = to_move
        scores.append(_score(tuple(child), other, me))
    return max(scores) if to_move == me else min(scores)


class MinimaxEngine:
    name = "minimax"

    def __init__(self, mark: str = "O", strength: str = "perfect", seed: int | None = None):
        self.mark = mark
        self.strength = strength
        self._rng = random.Random(seed)

    def decide(self, board: list[str], history: list[tuple[str, int]]) -> int:
        legal = _game.legal_moves(board)
        if not legal:
            raise ValueError("no legal moves")
        if self.strength == "random":
            return self._rng.choice(legal)
        other = "X" if self.mark == "O" else "O"
        best, best_score = None, -2
        for m in legal:
            child = list(board)
            child[m] = self.mark
            s = _score(tuple(child), other, self.mark)
            if s > best_score:
                best, best_score = m, s
        return best
