"""Motion gate — mean abs diff vs the previous frame.

This is what handles the hand drawing: frames with movement are gated and
never reach the reader. Known false-trigger: flickering lighting aliased at
low FPS; the debug banner prints the live diff so it's diagnosable in
seconds (fix: lock camera exposure, or raise T_MOTION via env).
"""

import numpy as np


class MotionGate:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def update(self, gray: np.ndarray) -> tuple[bool, float]:
        """Returns (motion, diff). First frame is treated as still."""
        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return False, 0.0
        diff = float(np.mean(np.abs(gray.astype(np.int16) - self._prev.astype(np.int16))))
        self._prev = gray
        return diff > self.threshold, diff
