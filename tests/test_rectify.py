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


def test_locks_on_strokes_not_paper_edges():
    """Regression: paper on a dark surface. The paper boundary produces the
    longest, strongest lines in the frame; the lock must land on the drawn
    ink, which is dark with bright paper on both sides."""
    img = np.full((480, 640), 55, np.uint8)          # dark desk
    cv2.rectangle(img, (140, 60), (560, 420), 215, -1)  # bright paper, hard edges
    cx, cy, third = 350, 240, 80
    for k in (0, 1):
        p = -third // 2 + k * third
        cv2.line(img, (cx + p, cy - 130), (cx + p, cy + 130), 45, 3, cv2.LINE_AA)
        cv2.line(img, (cx - 130, cy + p), (cx + 130, cy + p), 45, 3, cv2.LINE_AA)
    m = int(third * 0.28)
    cv2.line(img, (cx - m, cy - m), (cx + m, cy + m), 30, 4, cv2.LINE_AA)
    cv2.line(img, (cx + m, cy - m), (cx - m, cy + m), 30, 4, cv2.LINE_AA)

    fit = rectify_grid.find_grid(img)
    assert fit is not None
    cells = read_cells(rectify_grid.warp(img, fit))
    assert cells[4].mark == "X"
    assert sum(1 for c in cells if c.mark) == 1      # nothing phantom


def test_faint_thin_strokes_on_tilted_paper_with_textured_desk():
    """Regression: thin ballpoint (~2px, low contrast) on a tilted page over
    textured dark leather. Both prior failure modes at once: the strokes must
    still register as lines, and the tilted paper edges must lose."""
    rng = np.random.default_rng(3)
    img = (55 + rng.integers(-18, 18, (480, 640))).clip(0, 255).astype(np.uint8)
    paper = np.array([[150, 90], [590, 60], [610, 400], [130, 430]], np.int32)
    cv2.fillPoly(img, [paper], 212)
    cx, cy, third = 370, 240, 85
    for k in (0, 1):
        p = -third // 2 + k * third
        cv2.line(img, (cx + p + 6, cy - 130), (cx + p, cy + 130), 150, 2, cv2.LINE_AA)
        cv2.line(img, (cx - 130, cy + p), (cx + 130, cy + p - 5), 150, 2, cv2.LINE_AA)
    m = int(third * 0.28)
    cv2.line(img, (cx - m, cy - m), (cx + m, cy + m), 140, 3, cv2.LINE_AA)
    cv2.line(img, (cx + m, cy - m), (cx - m, cy + m), 140, 3, cv2.LINE_AA)

    fit = rectify_grid.find_grid(img)
    assert fit is not None, "no lock on faint strokes"
    # the lock must be the drawn grid, not the page: center-cell corners sit
    # near the true intersections, whose pairwise spread is one cell (~85px)
    quad = np.array(fit.corners)
    span = max(quad[:, 0].max() - quad[:, 0].min(),
               quad[:, 1].max() - quad[:, 1].min())
    assert span < 400, f"lock spans {span:.0f}px — that's the paper, not the grid"
    cells = read_cells(rectify_grid.warp(img, fit))
    assert cells[4].mark == "X"


def test_quad_sanity_rejects_slivers():
    """A page mid-motion can yield near-collinear intersections — a sliver
    quad must never become a homography."""
    square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)
    assert rectify_grid._quad_ok(square, 100, 100, 480, 640)
    rot = np.array([[150, 80], [220, 150], [150, 220], [80, 150]], float)
    assert rectify_grid._quad_ok(rot, 100, 100, 480, 640)
    # a sliver comes with wildly asymmetric family spacings
    sliver = np.array([[100, 100], [600, 110], [605, 122], [110, 115]], float)
    assert not rectify_grid._quad_ok(sliver, 500, 13, 480, 640)
    # and a shear-collapsed quad fails on corner angles
    shear = np.array([[100, 100], [200, 100], [290, 120], [190, 120]], float)
    assert not rectify_grid._quad_ok(shear, 100, 100, 480, 640)
    offscreen = square + [2000, 0]
    assert not rectify_grid._quad_ok(offscreen, 100, 100, 480, 640)


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
