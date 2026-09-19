"""Change detection for submarine game - monitors bounded area for image changes."""

import cv2
import numpy as np


class ChangeDetector:
    """Detects when a bounded area has settled (no changes for N frames)."""

    def __init__(self, settle_frames: int = 8, threshold: float = 0.05):
        """
        Args:
            settle_frames: Number of stable frames required
            threshold: Change threshold (0-1), lower = more sensitive
        """
        self.settle_frames = settle_frames
        self.threshold = threshold
        self.prev_frame = None
        self.stable_count = 0
        self.last_hash = None

    def check_change(self, frame: np.ndarray, corners: list[tuple[int, int]] = None) -> dict:
        """Check if frame has changed significantly.

        Args:
            frame: Current frame (grayscale or color)
            corners: Optional list of 4 corner points [(x,y), ...] to bound the area
                    If None, uses entire frame

        Returns:
            {
                'changed': bool,
                'settled': bool,  # True if stable for settle_frames
                'stable_count': int,
                'diff_score': float
            }
        """
        # Extract bounded region if corners provided
        if corners and len(corners) == 4:
            # Simple bounding box approach (could use perspective transform for accuracy)
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Compute perceptual hash for quick comparison
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        curr_hash = (small > small.mean()).astype(np.uint8).tobytes()

        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            self.last_hash = curr_hash
            return {
                'changed': False,
                'settled': False,
                'stable_count': 0,
                'diff_score': 0.0
            }

        # Compute frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        mean_diff = diff.mean() / 255.0  # Normalize to 0-1

        # Check if changed
        changed = mean_diff > self.threshold or curr_hash != self.last_hash

        if not changed:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.prev_frame = gray.copy()
            self.last_hash = curr_hash

        settled = self.stable_count >= self.settle_frames

        return {
            'changed': changed,
            'settled': settled,
            'stable_count': self.stable_count,
            'diff_score': mean_diff
        }

    def reset(self):
        """Reset detector state."""
        self.prev_frame = None
        self.stable_count = 0
        self.last_hash = None
