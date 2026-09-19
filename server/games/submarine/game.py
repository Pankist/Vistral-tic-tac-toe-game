"""Submarine puzzle game — vision-based logical puzzle solver.

Unlike turn-based games, submarine watches for puzzles, recognizes them,
solves them, and displays answers. Board state tracks the current puzzle.
"""


class Submarine:
    """Puzzle solver game - watches camera for logical challenges."""

    name = "submarine"
    rows = 3  # Display grid for compatibility
    cols = 3
    human_mark = ""  # Not applicable
    agent_mark = "✓"
    marks = ("✓",)

    def initial_board(self) -> dict:
        """Board is a puzzle state dict, not a cell grid."""
        return {
            "puzzle_text": "",
            "answer": "",
            "status": "monitoring",  # monitoring | settled | recognizing | solved | error
            "raw_image": None,
        }

    def legal_moves(self, board: dict) -> list[int]:
        """No traditional moves - return [0] if ready to solve."""
        if board["puzzle_text"] and board["status"] == "recognizing":
            return [0]
        return []

    def apply(self, board: dict, move: int, mark: str) -> dict:
        """Apply doesn't really apply here - used for state transitions."""
        return board

    def winner(self, board: dict) -> str | None:
        """No winner concept - puzzle is 'won' when solved."""
        if board["status"] == "solved" and board["answer"]:
            return "✓"
        return None

    def win_line(self, board: dict) -> tuple | None:
        return None

    def is_draw(self, board: dict) -> bool:
        """Puzzle with no solution detected."""
        return board["status"] == "error"

    def rules_text(self) -> str:
        return (
            "Submarine puzzle solver: The system monitors a bounded camera area "
            "for logical puzzles, IQ test questions, or challenges. When the image "
            "settles, it recognizes the puzzle using vision AI, solves it, and "
            "displays the answer."
        )

    def serialize(self, board: dict) -> str:
        """Format puzzle state for display."""
        if board["puzzle_text"]:
            return f"Puzzle: {board['puzzle_text']}\nAnswer: {board.get('answer', '...')}"
        return "Monitoring for puzzles..."

    def describe_cell(self, move: int) -> str:
        return "puzzle"
