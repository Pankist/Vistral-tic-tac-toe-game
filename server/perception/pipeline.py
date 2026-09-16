"""Per-frame orchestration: decode -> motion gate -> rectify -> read cells.

Stateless per frame except the motion gate's previous-frame buffer and the
orientation hint. Temporal debounce is the FSM's job, not ours. Target
≤40ms/frame on laptop CPU at 4 FPS.

Orientation: a square '#' is 4-fold ambiguous. Of the 4 candidate rotations
we keep the one consistent with the last confirmed board (the hint comes from
the session; geometry stays stateless). Before any mark exists every rotation
matches, so the lock orientation is simply adopted as the agent's frame.
"""

import time

import cv2
import numpy as np

from server.core import config as cfg
from server.perception import debug as debug_mod
from server.perception import rectify_aruco, rectify_grid
from server.perception.motion import MotionGate
from server.perception.types import PerceptionResult

_ROT_IDX = [np.rot90(np.arange(9).reshape(3, 3), k).flatten().tolist() for k in range(4)]


class Pipeline:
    def __init__(self, marks_module):
        self.marks = marks_module
        self.gate = MotionGate(cfg.T_MOTION)
        self.hint: list[str] | None = None   # last confirmed board labels
        self.last_k = 0                       # last chosen rotation
        self.last_rectified: np.ndarray | None = None
        self.last_ms = 0.0
        self.t_empty_override: float | None = None  # memory-learned, clamped upstream

    def process(self, jpeg: bytes, banner: str = "") -> tuple[PerceptionResult, bytes]:
        t0 = time.perf_counter()
        arr = np.frombuffer(jpeg, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        ts = time.time()
        if frame is None:
            return PerceptionResult(False, None, None, [], False, ts), b""
        if frame.shape[1] > cfg.FRAME_WIDTH:
            scale = cfg.FRAME_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, int(frame.shape[0] * scale)))

        motion, diff = self.gate.update(frame)
        if motion:
            result = PerceptionResult(False, None, None, [], True, ts, diff=round(diff, 1))
            dbg = debug_mod.composite(frame, None, None, None,
                                      f"{banner} | diff {diff:.1f}/{cfg.T_MOTION} GATED")
            self.last_ms = (time.perf_counter() - t0) * 1000
            return result, dbg

        fit, method = self._rectify(frame)
        if fit is None:
            result = PerceptionResult(False, None, None, [], False, ts, diff=round(diff, 1))
            dbg = debug_mod.composite(frame, None, None, None,
                                      f"{banner} | diff {diff:.1f}/{cfg.T_MOTION} | no grid")
            self.last_ms = (time.perf_counter() - t0) * 1000
            return result, dbg

        rectified = rectify_grid.warp(frame, fit, cfg.CANONICAL)
        cells = self.marks.read_cells(rectified, self.t_empty_override)
        cells, rectified, k = self._orient(cells, rectified)

        result = PerceptionResult(
            grid_found=True, corners=fit.corners, reproj_error=fit.reproj_error,
            cells=cells, motion=False, ts=ts, diff=round(diff, 1), method=method,
        )
        self.last_rectified = rectified
        self.last_ms = (time.perf_counter() - t0) * 1000
        dbg = debug_mod.composite(
            frame, rectified, cells, fit.corners,
            f"{banner} | diff {diff:.1f}/{cfg.T_MOTION} | {method} lock "
            f"reproj {fit.reproj_error} | {self.last_ms:.0f}ms",
        )
        return result, dbg

    def _rectify(self, frame):
        mode = cfg.RECTIFY
        if mode == "aruco" or (mode == "auto" and rectify_aruco.markers_present(frame)):
            fit = rectify_aruco.find_grid(frame, cfg.CANONICAL)
            if fit is not None:
                return fit, "aruco"
            if mode == "aruco":
                return None, ""
        return rectify_grid.find_grid(frame, cfg.CANONICAL), "lines"

    def _orient(self, cells, rectified):
        """Pick the rotation of the 4 candidates that best matches the hint."""
        if not self.hint or all(m == "" for m in self.hint):
            return cells, rectified, self.last_k
        best_k, best_score = self.last_k, -1
        for k in range(4):
            perm = _ROT_IDX[k]
            score = sum(
                1 for i in range(9)
                if self.hint[i] != "" and cells[perm[i]].mark == self.hint[i]
            )
            # stability preference: keep the previous rotation on ties
            if score > best_score or (score == best_score and k == self.last_k):
                best_k, best_score = k, score
        perm = _ROT_IDX[best_k]
        cells = [cells[perm[i]] for i in range(9)]
        rectified = np.ascontiguousarray(np.rot90(rectified, best_k))
        self.last_k = best_k
        return cells, rectified, best_k
