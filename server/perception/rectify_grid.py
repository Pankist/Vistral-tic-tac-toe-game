"""Markerless, borderless grid detection [B — the tuned hot spot].

The board is a plain hand-drawn '#': no outer border, no fiducials.
Canny -> HoughLinesP segments -> cluster by angle into two roughly-orthogonal
line families (rotation-invariant: any two directions ~90 degrees apart) ->
fit the 2 dominant lines per family -> their 4 intersections are the center
cell's corners -> homography maps them to the canonical inner square; outer
cells are extrapolated congruent to the center cell.

Stated assumption: marks are drawn near the intersections; ink far beyond the
stroke ends is out of spec. Reject if families aren't 2+2, spacing is
implausible, or the fit residual is high — a rejected frame is just
"no board, keep looking", never a wrong lock.
"""

from dataclasses import dataclass

import cv2
import numpy as np

ANGLE_TOL = np.deg2rad(25)     # how far a segment may sit from its family mean
GAP_FRAC = 0.04                # 1-D offset gap that separates two grid lines
MIN_SPACING = 0.10             # min |line1-line2| offset, fraction of min(h,w)
MAX_SPACING = 0.70


@dataclass
class GridFit:
    H: np.ndarray                  # frame -> canonical homography
    corners: list[list[float]]     # full-board quad in frame coords, TL TR BR BL
    reproj_error: float


def _ang_dist(a: float, b: float) -> float:
    """Distance between undirected line angles, mod pi."""
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _cluster_offsets(offsets: np.ndarray, weights: np.ndarray, gap: float):
    """1-D clustering by sorted gap; returns [(weighted_center, total_weight, member_idx)]."""
    order = np.argsort(offsets)
    clusters, current = [], [order[0]]
    for i in order[1:]:
        if offsets[i] - offsets[current[-1]] > gap:
            clusters.append(current)
            current = [i]
        else:
            current.append(i)
    clusters.append(current)
    out = []
    for members in clusters:
        w = weights[members]
        out.append((float(np.average(offsets[members], weights=w)), float(w.sum()), members))
    return out


def _fit_family_line(segs: np.ndarray, members: list[int]) -> np.ndarray:
    """Homogeneous line through all endpoints of the member segments."""
    pts = np.vstack([segs[members][:, :2], segs[members][:, 2:]]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    # line as ax + by + c = 0 with normal (a,b) = (-vy, vx)
    return np.array([-vy, vx, vy * x0 - vx * y0], dtype=np.float64)


def _residual(line: np.ndarray, segs: np.ndarray, members: list[int]) -> float:
    pts = np.vstack([segs[members][:, :2], segs[members][:, 2:]])
    a, b, c = line
    return float(np.mean(np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / np.hypot(a, b)))


def find_grid(gray: np.ndarray, canonical: int = 330) -> GridFit | None:
    h, w = gray.shape
    short = min(h, w)
    edges = cv2.Canny(gray, 50, 150)
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40,
        minLineLength=int(short * 0.20), maxLineGap=int(short * 0.03),
    )
    if segs is None or len(segs) < 4:
        return None
    segs = segs.reshape(-1, 4).astype(np.float64)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx) % np.pi

    # two orthogonal families, seeded by the longest segment's direction
    theta0 = angles[np.argmax(lengths)]
    fam_a = [i for i in range(len(segs)) if _ang_dist(angles[i], theta0) < ANGLE_TOL]
    fam_b = [i for i in range(len(segs))
             if _ang_dist(angles[i], theta0 + np.pi / 2) < ANGLE_TOL]
    if len(fam_a) < 2 or len(fam_b) < 2:
        return None

    lines, total_residual = [], 0.0
    for fam in (fam_a, fam_b):
        idx = np.array(fam)
        # undirected angles: shift each member to the branch nearest the seed
        seed = theta0 if fam is fam_a else (theta0 + np.pi / 2) % np.pi
        branch = angles[idx].copy()
        branch[branch - seed > np.pi / 2] -= np.pi
        branch[seed - branch > np.pi / 2] += np.pi
        fam_theta = float(np.average(branch, weights=lengths[idx]))
        normal = np.array([-np.sin(fam_theta), np.cos(fam_theta)])
        mids = np.stack([(segs[idx, 0] + segs[idx, 2]) / 2,
                         (segs[idx, 1] + segs[idx, 3]) / 2], axis=1)
        offsets = mids @ normal
        clusters = _cluster_offsets(offsets, lengths[idx], gap=short * GAP_FRAC)
        if len(clusters) < 2:
            return None
        clusters.sort(key=lambda c: -c[1])
        top2 = sorted(clusters[:2], key=lambda c: c[0])
        if not (MIN_SPACING * short < abs(top2[1][0] - top2[0][0]) < MAX_SPACING * short):
            return None
        for _, _, members in top2:
            line = _fit_family_line(segs, list(idx[members]))
            total_residual += _residual(line, segs, list(idx[members]))
            lines.append(line)

    # 4 intersections of the 2x2 dominant lines = center cell corners
    pts = []
    for la in lines[:2]:
        for lb in lines[2:]:
            p = np.cross(la, lb)
            if abs(p[2]) < 1e-9:
                return None
            pts.append(p[:2] / p[2])
    pts = np.array(pts)

    # order TL TR BR BL by angle around the centroid, then roll TL first
    centroid = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
    quad = pts[order]
    tl = np.argmin(quad.sum(axis=1))
    quad = np.roll(quad, -tl, axis=0)

    third = canonical / 3
    inner = np.array([[third, third], [2 * third, third],
                      [2 * third, 2 * third], [third, 2 * third]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), inner)

    # full-board quad back in frame coords, for the client overlay
    Hinv = np.linalg.inv(H)
    board = np.array([[0, 0], [canonical, 0], [canonical, canonical], [0, canonical]],
                     dtype=np.float64)
    board_h = np.hstack([board, np.ones((4, 1))]) @ Hinv.T
    corners = (board_h[:, :2] / board_h[:, 2:3]).tolist()

    return GridFit(H=H, corners=corners, reproj_error=round(total_residual / 4, 2))


def warp(gray: np.ndarray, fit: GridFit, canonical: int = 330) -> np.ndarray:
    return cv2.warpPerspective(gray, fit.H, (canonical, canonical),
                               flags=cv2.INTER_AREA, borderValue=255)
