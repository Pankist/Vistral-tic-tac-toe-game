import pytest

from server.games.tic_tac_toe.game import WIN_LINES, TicTacToe

g = TicTacToe()


def board(s: str) -> list[str]:
    """'X.O......' -> board list"""
    return ["" if c == "." else c for c in s]


def test_initial_board_empty():
    assert g.initial_board() == [""] * 9
    assert g.legal_moves(g.initial_board()) == list(range(9))


def test_apply_and_legality():
    b = g.apply(g.initial_board(), 4, "X")
    assert b[4] == "X"
    assert 4 not in g.legal_moves(b)
    with pytest.raises(ValueError):
        g.apply(b, 4, "O")
    with pytest.raises(ValueError):
        g.apply(b, 0, "Z")


@pytest.mark.parametrize("line", WIN_LINES)
@pytest.mark.parametrize("mark", ["X", "O"])
def test_all_win_lines_detected(line, mark):
    b = [""] * 9
    for c in line:
        b[c] = mark
    assert g.winner(b) == mark
    assert g.win_line(b) == line


def test_no_winner_and_draw():
    assert g.winner(board("XOXOXOOXO")) is None
    assert g.is_draw(board("XOXOXOOXO"))
    assert not g.is_draw(board("X........"))


def test_win_is_not_draw():
    b = board("XXXOO....")
    assert g.winner(b) == "X"
    assert not g.is_draw(b)


def test_serialize_readable():
    s = g.serialize(board("X...O...X"))
    assert s.splitlines() == ["X | . | .", ". | O | .", ". | . | X"]


def test_cell_names():
    assert g.describe_cell(0) == "top-left"
    assert g.describe_cell(4) == "center"
    assert g.describe_cell(8) == "bottom-right"
