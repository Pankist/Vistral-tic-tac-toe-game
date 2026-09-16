"""3x3 cell split + X/O classification — game-specific perception.

This lives in the game package because reading X vs O is game semantics;
perception/ owns geometry only and calls in here through the pipeline.

Classification [B — tuned on-site]:
  O = outer contour with a child hole and circularity > O_CIRCULARITY
  X = ink with no enclosed hole (two crossing strokes merge into one component)
Confidence is the margin over the deciding thresholds; anything ambiguous
comes out low-conf, which is arbiter territory, never a guess.
"""

import cv2
import numpy as np

from server.games.tic_tac_toe import config as gcfg
from server.perception.types import CellRead


def read_cells(rect_gray: np.ndarray, t_empty: float | None = None) -> list[CellRead]:
    """rect_gray: canonical CANONICAL x CANONICAL grayscale board."""
    t_empty = gcfg.T_EMPTY if t_empty is None else t_empty
    side = rect_gray.shape[0]
    cell = side // 3
    margin = int(cell * gcfg.CELL_MARGIN)
    out: list[CellRead] = []
    for r in range(3):
        for c in range(3):
            y0, x0 = r * cell + margin, c * cell + margin
            crop = rect_gray[y0:y0 + cell - 2 * margin, x0:x0 + cell - 2 * margin]
            out.append(_read_one(crop, t_empty))
    return out


def _read_one(crop: np.ndarray, t_empty: float) -> CellRead:
    # local, not global, binarization: absorbs brightness swings one layer down
    binary = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    area = binary.size
    ink = int(np.count_nonzero(binary))
    ink_ratio = ink / area

    if ink_ratio < t_empty:
        # confidence grows as ink falls further below the cutoff
        conf = min(1.0, 0.6 + (t_empty - ink_ratio) / t_empty * 0.4)
        return CellRead(mark="", conf=round(conf, 2), ink_ratio=round(ink_ratio, 3))

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CellRead(mark="", conf=0.3, ink_ratio=round(ink_ratio, 3))

    # dominant outer contour and its largest child hole
    outer_idx = max(
        (i for i in range(len(contours)) if hierarchy[0][i][3] == -1),
        key=lambda i: cv2.contourArea(contours[i]),
    )
    outer = contours[outer_idx]
    outer_area = cv2.contourArea(outer)
    perimeter = cv2.arcLength(outer, True)
    circularity = 4 * np.pi * outer_area / (perimeter * perimeter) if perimeter > 0 else 0.0

    hole_area = 0.0
    child = hierarchy[0][outer_idx][2]
    while child != -1:
        hole_area = max(hole_area, cv2.contourArea(contours[child]))
        child = hierarchy[0][child][0]
    has_hole = hole_area / area > gcfg.MIN_HOLE_RATIO

    if has_hole and circularity > gcfg.O_CIRCULARITY:
        margin = min(1.0, (circularity - gcfg.O_CIRCULARITY) / (1 - gcfg.O_CIRCULARITY))
        conf = 0.7 + 0.3 * margin
        return CellRead(mark="O", conf=round(conf, 2), ink_ratio=round(ink_ratio, 3))
    if not has_hole:
        # the further from O-like circularity, the surer we are it's an X
        conf = 0.6 + 0.4 * min(1.0, max(0.0, (gcfg.O_CIRCULARITY - circularity) / gcfg.O_CIRCULARITY))
        return CellRead(mark="X", conf=round(conf, 2), ink_ratio=round(ink_ratio, 3))
    # hole but not circular, or circular but no hole: ambiguous on purpose
    guess = "O" if circularity > gcfg.O_CIRCULARITY else "X"
    return CellRead(mark=guess, conf=0.5, ink_ratio=round(ink_ratio, 3))
