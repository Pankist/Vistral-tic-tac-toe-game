"""Perception contract — what the pipeline hands the FSM, per frame."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class CellRead:
    mark: Literal["X", "O", ""]
    conf: float          # 0..1
    ink_ratio: float


@dataclass
class PerceptionResult:
    grid_found: bool
    corners: list | None            # 4 [x,y] points, ordered TL TR BR BL, frame coords
    reproj_error: float | None
    cells: list[CellRead]           # len 9, row-major; empty list if no grid
    motion: bool                    # True = hand/movement, frame gated
    ts: float
    diff: float = 0.0               # motion-gate diff value, for the debug banner
    method: str = ""                # "lines" | "aruco" | ""

    def labels(self) -> tuple[str, ...]:
        """Symbolic hypothesis: identity is labels, not pixels or confidence."""
        return tuple(c.mark for c in self.cells)
