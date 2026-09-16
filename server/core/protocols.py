"""The contracts. core/ and perception/ never import a concrete game except
through these. A move is an int cell index; board is a game-defined list of
mark strings. Only methods tic_tac_toe actually uses today — nothing added
"for chess"; the empty chess package marks the seam.
"""

from typing import Protocol

Board = list[str]
Move = int


class Game(Protocol):
    name: str
    rows: int
    cols: int
    human_mark: str
    agent_mark: str
    marks: tuple[str, ...]

    def initial_board(self) -> Board: ...
    def legal_moves(self, board: Board) -> list[Move]: ...
    def apply(self, board: Board, move: Move, mark: str) -> Board: ...
    def winner(self, board: Board) -> str | None: ...
    def is_draw(self, board: Board) -> bool: ...
    def rules_text(self) -> str: ...
    def serialize(self, board: Board) -> str: ...
    def describe_cell(self, move: Move) -> str: ...


class Engine(Protocol):
    name: str

    def decide(self, board: Board, history: list[tuple[str, Move]]) -> Move: ...
