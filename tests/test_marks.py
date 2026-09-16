"""X/O classification on synthetic drawn cells — no camera needed."""

import cv2
import numpy as np

from server.games.tic_tac_toe.marks import read_cells


def blank_board(side=330):
    return np.full((side, side), 235, np.uint8)


def draw_x(img, cell, thickness=7):
    r, c = divmod(cell, 3)
    s = img.shape[0] // 3
    x0, y0 = c * s, r * s
    m = int(s * 0.22)
    cv2.line(img, (x0 + m, y0 + m), (x0 + s - m, y0 + s - m), 20, thickness)
    cv2.line(img, (x0 + s - m, y0 + m), (x0 + m, y0 + s - m), 20, thickness)


def draw_o(img, cell, thickness=7):
    r, c = divmod(cell, 3)
    s = img.shape[0] // 3
    cv2.circle(img, (c * s + s // 2, r * s + s // 2), int(s * 0.28), 20, thickness)


def test_empty_board_reads_empty():
    cells = read_cells(blank_board())
    assert all(c.mark == "" for c in cells)
    assert all(c.ink_ratio < 0.03 for c in cells)


def test_x_and_o_classified():
    img = blank_board()
    draw_x(img, 0)
    draw_o(img, 4)
    draw_x(img, 8)
    cells = read_cells(img)
    assert cells[0].mark == "X" and cells[0].conf > 0.6
    assert cells[4].mark == "O" and cells[4].conf > 0.6
    assert cells[8].mark == "X"
    assert all(cells[i].mark == "" for i in (1, 2, 3, 5, 6, 7))


def test_full_board_mixed():
    img = blank_board()
    layout = ["X", "O", "X", "O", "X", "O", "X", "O", "X"]
    for i, m in enumerate(layout):
        (draw_x if m == "X" else draw_o)(img, i)
    cells = read_cells(img)
    assert [c.mark for c in cells] == layout


def test_small_thin_o_is_not_an_x():
    """Regression: a small O drawn with a thin pen. Its hole is tiny relative
    to the cell and its ring may not survive denoising — it must still never
    read as a confident X."""
    img = blank_board()
    r, c = divmod(2, 3)
    s = img.shape[0] // 3
    cv2.circle(img, (c * s + s // 2, r * s + s // 2), 13, 20, 3)
    cell = read_cells(img)[2]
    assert cell.mark == "O", f"read {cell.mark} conf {cell.conf}"


def test_broken_ring_o_reads_o_at_low_confidence():
    """An O whose ring has a gap: no enclosed hole exists at all. Center
    occupancy must still call it an O, at arbiter-worthy confidence."""
    img = blank_board()
    cv2.ellipse(img, (165, 165), (30, 30), 0, 20, 330, 20, 4)   # 310° arc
    cell = read_cells(img)[4]
    assert cell.mark == "O"
    assert cell.conf <= 0.75          # asks for a closer look, not a confirm


def test_learned_t_empty_override_applies():
    img = blank_board()
    # smudge below default threshold but above a stricter learned one
    cv2.circle(img, (55, 55), 10, 120, 2)
    default_read = read_cells(img)[0]
    strict_read = read_cells(img, t_empty=0.001)[0]
    assert default_read.mark == ""
    assert strict_read.mark != "" or strict_read.conf <= 0.6
