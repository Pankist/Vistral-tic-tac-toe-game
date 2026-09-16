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


def _quad_ok(quad: np.ndarray, sa: float, sb: float, h: int, w: int) -> bool:
    """The center cell must be a plausible quadrilateral before it becomes a
    homography. A page mid-motion (lifted, curled, half out of frame) can
    yield a 2+2 line fit whose intersections are nearly collinear — a sliver
    quad that smears ink across every cell. Reject on area vs the measured
    line spacings, convexity, corner angles, and being wildly out of frame."""
    if (quad[:, 0].min() < -0.5 * w or quad[:, 0].max() > 1.5 * w
            or quad[:, 1].min() < -0.5 * h or quad[:, 1].max() > 1.5 * h):
        return False
    q32 = quad.astype(np.float32)
    area = cv2.contourArea(q32)
    if not (0.4 * sa * sb <= area <= 2.5 * sa * sb):
        return False
    if not cv2.isContourConvex(q32):
        return False
    for i in range(4):
        v1 = quad[(i - 1) % 4] - quad[i]
        v2 = quad[(i + 1) % 4] - quad[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return False
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / denom, -1, 1)))
        if not 30 <= ang <= 150:
            return False
    return True


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


def _is_ink_stroke(gray: np.ndarray, line: np.ndarray, segs: np.ndarray,
                   members: list[int], margin: int = 12) -> bool:
    """A grid line is drawn ink: dark, with bright paper on BOTH sides, and
    the two sides are roughly EQUALLY bright.

    Both halves matter. Darker-than-both rejects most non-ink lines; the
    symmetry test is what reliably kills paper edges and shadow boundaries —
    paper on one side and dark desk on the other is a huge asymmetry no pen
    stroke produces. Probes are medians of 3 offsets per side, because a
    single textured-background pixel is too noisy to compare against.
    """
    h, w = gray.shape
    a, b, c = line
    norm = np.hypot(a, b)
    n = np.array([a, b]) / norm
    d = np.array([-b, a]) / norm
    p0 = -c / norm * n
    pts = np.vstack([segs[members][:, :2], segs[members][:, 2:]])
    ts = (pts - p0) @ d
    good = total = 0
    for t in np.linspace(ts.min(), ts.max(), 21):
        p = p0 + t * d

        def sample(sign: int) -> int | None:
            vals = []
            for probe in (7, 10, 13):
                x = int(round(p[0] + sign * probe * n[0]))
                y = int(round(p[1] + sign * probe * n[1]))
                if 0 <= x < w and 0 <= y < h:
                    vals.append(int(gray[y, x]))
            return int(np.median(vals)) if len(vals) == 3 else None

        x, y = int(round(p[0])), int(round(p[1]))
        if not (1 <= x < w - 1 and 1 <= y < h - 1):
            continue
        vp, vm = sample(+1), sample(-1)
        if vp is None or vm is None:
            continue
        total += 1
        v0 = int(gray[y - 1:y + 2, x - 1:x + 2].min())   # stroke center ± fit slack
        bright_both = min(vp, vm) - v0 >= margin
        symmetric = abs(vp - vm) <= 0.6 * (max(vp, vm) - v0)
        if bright_both and symmetric:
            good += 1
    return total >= 6 and good / total >= 0.55


def find_grid(gray: np.ndarray, canonical: int = 330) -> GridFit | None:
    h, w = gray.shape
    short = min(h, w)
    # Canny alone misses faint thin pen strokes at capture scale; union it with
    # the same local ink binarization the cell reader uses, so a ballpoint '#'
    # is as much a line source as a fat marker one.
    edges = cv2.Canny(gray, 40, 120)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    edges = cv2.bitwise_or(edges, ink)
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35,
        minLineLength=int(short * 0.16), maxLineGap=int(short * 0.04),
    )
    if segs is None or len(segs) < 4:
        return None
    segs = segs.reshape(-1, 4).astype(np.float64)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx) % np.pi

    # seed candidate family directions from the longest segments — but don't
    # trust any single one (the longest line in frame is often a paper edge,
    # not the grid); try distinct angles until a seed yields a valid 2+2 fit
    seeds, order = [], np.argsort(-lengths)
    for i in order:
        if all(_ang_dist(angles[i], s) > np.deg2rad(15)
               and _ang_dist(angles[i], s + np.pi / 2) > np.deg2rad(15)
               for s in seeds):
            seeds.append(float(angles[i]))
        if len(seeds) == 4:
            break
    for theta0 in seeds:
        fit = _try_seed(gray, segs, lengths, angles, theta0, short, canonical)
        if fit is not None:
            return fit
    return None


def _try_seed(gray, segs, lengths, angles, theta0, short, canonical):
    fam_a = [i for i in range(len(segs)) if _ang_dist(angles[i], theta0) < ANGLE_TOL]
    fam_b = [i for i in range(len(segs))
             if _ang_dist(angles[i], theta0 + np.pi / 2) < ANGLE_TOL]
    if len(fam_a) < 2 or len(fam_b) < 2:
        return None

    lines, total_residual, spacings = [], 0.0, []
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
        # fit every cluster and keep only real ink strokes (kills paper edges)
        fitted = []
        for center, weight, members in clusters:
            gidx = list(idx[members])
            line = _fit_family_line(segs, gidx)
            if _is_ink_stroke(gray, line, segs, gidx):
                fitted.append((center, weight, gidx, line))
        if len(fitted) < 2:
            return None
        fitted.sort(key=lambda f: -f[1])
        top2 = sorted(fitted[:2], key=lambda f: f[0])
        spacing = abs(top2[1][0] - top2[0][0])
        if not (MIN_SPACING * short < spacing < MAX_SPACING * short):
            return None
        spacings.append(spacing)
        for _, _, gidx, line in top2:
            total_residual += _residual(line, segs, gidx)
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
    h, w = gray.shape
    if not _quad_ok(quad, spacings[0], spacings[1], h, w):
        return None

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
