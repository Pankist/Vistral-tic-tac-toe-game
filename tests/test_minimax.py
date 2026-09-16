import random

from server.games.tic_tac_toe.engine_minimax import MinimaxEngine
from server.games.tic_tac_toe.game import TicTacToe

g = TicTacToe()


def play(engine_o, engine_x_move):
    """Play one full game; X moves via engine_x_move(board), O via engine_o."""
    b = g.initial_board()
    history = []
    while True:
        m = engine_x_move(b)
        b = g.apply(b, m, "X")
        history.append(("X", m))
        if g.winner(b) or g.is_draw(b):
            return g.winner(b)
        m = engine_o.decide(b, history)
        b = g.apply(b, m, "O")
        history.append(("O", m))
        if g.winner(b) or g.is_draw(b):
            return g.winner(b)


def test_minimax_never_loses_vs_random_500_games():
    rng = random.Random(42)
    engine = MinimaxEngine(mark="O", strength="perfect")
    losses = 0
    for _ in range(500):
        w = play(engine, lambda b: rng.choice(g.legal_moves(b)))
        if w == "X":
            losses += 1
    assert losses == 0


def test_minimax_takes_immediate_win():
    engine = MinimaxEngine(mark="O")
    b = ["O", "O", "", "X", "X", "", "", "", ""]
    assert engine.decide(b, []) == 2


def test_minimax_blocks_immediate_loss():
    engine = MinimaxEngine(mark="O")
    b = ["X", "X", "", "", "O", "", "", "", ""]
    assert engine.decide(b, []) == 2


def test_random_strength_is_legal_and_varied():
    engine = MinimaxEngine(mark="O", strength="random", seed=7)
    b = ["X", "", "", "", "", "", "", "", ""]
    moves = {engine.decide(b, []) for _ in range(50)}
    assert moves <= set(g.legal_moves(b))
    assert len(moves) > 1


def test_perfect_vs_perfect_is_draw():
    ex = MinimaxEngine(mark="X")
    eo = MinimaxEngine(mark="O")
    b = g.initial_board()
    turn = "X"
    while not g.winner(b) and not g.is_draw(b):
        e = ex if turn == "X" else eo
        b = g.apply(b, e.decide(b, []), turn)
        turn = "O" if turn == "X" else "X"
    assert g.winner(b) is None and g.is_draw(b)
