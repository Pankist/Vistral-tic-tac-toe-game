"""Markerless grid detection on synthetic frames — drawn '#', rotation,
perspective. Real-lighting tuning happens live; this pins the geometry."""

import cv2
import numpy as np

from server.perception import rectify_grid
from server.games.tic_tac_toe.marks import read_cells


def draw_hash(angle_deg=0.0, offset=(0, 0), size=480):
    """A thick hand-drawn-ish '#' on paper, optionally rotated."""
    img = np.full((size, int(size * 4 / 3)), 210, np.uint8)
    h, w = img.shape
    cx, cy = w // 2 + offset[0], h // 2 + offset[1]
    span, third = int(size * 0.55), int(size * 0.55) // 3
    lines = []
    for i in (0, 1):
        p = -third // 2 + i * third
        lines.append(((cx + p, cy - span // 2), (cx + p, cy + span // 2)))
        lines.append(((cx - span // 2, cy + p), (cx + span // 2, cy + p)))
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    def rot(pt):
        v = M @ np.array([pt[0], pt[1], 1.0])
        return int(v[0]), int(v[1])

    for a, b in lines:
        cv2.line(img, rot(a), rot(b), 45, 6, cv2.LINE_AA)
    return img, (cx, cy), third, M


def test_finds_axis_aligned_grid():
    img, _, _, _ = draw_hash()
    fit = rectify_grid.find_grid(img)
    assert fit is not None
    # residual is stroke-width-scaled (Hough sees both edges of a 6px stroke)
    assert fit.reproj_error < 6.0


def test_finds_rotated_grid():
    for angle in (10, 30, 45):
        img, _, _, _ = draw_hash(angle_deg=angle)
        assert rectify_grid.find_grid(img) is not None, f"angle {angle}"


def test_rejects_blank_frame():
    img = np.full((480, 640), 210, np.uint8)
    assert rectify_grid.find_grid(img) is None


def test_rejects_single_line():
    img = np.full((480, 640), 210, np.uint8)
    cv2.line(img, (100, 240), (540, 240), 45, 6)
    assert rectify_grid.find_grid(img) is None


def test_end_to_end_marks_through_rectification():
    img, (cx, cy), third, M = draw_hash(angle_deg=8)

    def rot(pt):
        v = M @ np.array([pt[0], pt[1], 1.0])
        return int(v[0]), int(v[1])

    # X in the center cell, O in the cell above it (drawn pre-rotation coords)
    x0, y0 = cx, cy
    m = int(third * 0.28)
    cv2.line(img, rot((x0 - m, y0 - m)), rot((x0 + m, y0 + m)), 25, 6, cv2.LINE_AA)
    cv2.line(img, rot((x0 + m, y0 - m)), rot((x0 - m, y0 + m)), 25, 6, cv2.LINE_AA)
    cv2.circle(img, rot((cx, cy - third)), int(third * 0.3), 25, 6, cv2.LINE_AA)

    fit = rectify_grid.find_grid(img)
    assert fit is not None
    rect = rectify_grid.warp(img, fit)
    cells = read_cells(rect)
    assert cells[4].mark == "X"
    assert cells[1].mark == "O"
    marked = {i for i, c in enumerate(cells) if c.mark}
    assert marked == {1, 4}
