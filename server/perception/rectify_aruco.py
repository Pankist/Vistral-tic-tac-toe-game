"""Fiducial insurance path — placeholder, deliberately not implemented.

The markerless '#' detector (rectify_grid.py) is the one and only shipping
path. This module marks the seam where a fiducial fallback would go if a
venue's lighting ever defeats the line detector: printed 4x4_50 ArUco
markers, ids 0-3 at TL TR BR BL of the grid area, homography from marker
centers, orientation resolved by id (killing the 4-fold ambiguity of a bare
'#'). Config already routes here (RECTIFY="aruco"); wiring it up is an
isolated change to this file plus a printed page.
"""

import numpy as np

from server.perception.rectify_grid import GridFit


def find_grid(gray: np.ndarray, canonical: int = 330) -> GridFit | None:
    """Placeholder: no fiducial detection. Always reports no grid."""
    return None


def markers_present(gray: np.ndarray) -> bool:
    """Placeholder: RECTIFY="auto" therefore always uses the line detector."""
    return False
