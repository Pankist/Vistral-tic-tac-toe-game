"""Board state, legality, win/draw — implements core.protocols.Game.
Board is a list of 9 strings ("X" | "O" | ""), row-major, cell = row*3+col.
"""

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)

_CELL_NAMES = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)


class TicTacToe:
    name = "tic_tac_toe"
    rows = 3
    cols = 3
    human_mark = "X"
    agent_mark = "O"
    marks = ("X", "O")

    def initial_board(self) -> list[str]:
        return [""] * 9

    def legal_moves(self, board: list[str]) -> list[int]:
        return [i for i, c in enumerate(board) if c == ""]

    def apply(self, board: list[str], move: int, mark: str) -> list[str]:
        if board[move] != "":
            raise ValueError(f"cell {move} is occupied")
        if mark not in self.marks:
            raise ValueError(f"unknown mark {mark!r}")
        out = list(board)
        out[move] = mark
        return out

    def winner(self, board: list[str]) -> str | None:
        for a, b, c in WIN_LINES:
            if board[a] != "" and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def win_line(self, board: list[str]) -> tuple[int, int, int] | None:
        for line in WIN_LINES:
            a, b, c = line
            if board[a] != "" and board[a] == board[b] == board[c]:
                return line
        return None

    def is_draw(self, board: list[str]) -> bool:
        return self.winner(board) is None and all(c != "" for c in board)

    def rules_text(self) -> str:
        return (
            "Tic-tac-toe on a 3x3 grid. Cells are numbered 0-8, row-major "
            "(0 top-left, 4 center, 8 bottom-right). Players alternate turns "
            "placing their mark in an empty cell. Three of the same mark in a "
            "row, column, or diagonal wins. A full board with no winner is a draw."
        )

    def serialize(self, board: list[str]) -> str:
        rows = []
        for r in range(3):
            rows.append(" | ".join(board[r * 3 + c] or "." for c in range(3)))
        return "\n".join(rows)

    def describe_cell(self, move: int) -> str:
        return _CELL_NAMES[move]
