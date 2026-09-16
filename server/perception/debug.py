"""Debug composite — what the agent sees, streamed to the client every
processed frame. 2-up raw + rectified, per-cell overlay, state banner.
This doubles as the projector view; keep it legible from across a room.
"""

import cv2
import numpy as np

BANNER_H = 26
FONT = cv2.FONT_HERSHEY_SIMPLEX


def composite(
    raw_gray: np.ndarray,
    rectified: np.ndarray | None,
    cells,                      # list[CellRead] | None
    corners,                    # frame-coord quad | None
    banner: str,
    jpeg_quality: int = 70,
) -> bytes:
    view_h = 330
    raw = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
    scale = view_h / raw.shape[0]
    raw = cv2.resize(raw, (int(raw.shape[1] * scale), view_h))
    if corners:
        pts = (np.array(corners) * scale).astype(np.int32)
        cv2.polylines(raw, [pts], True, (163, 201, 53), 2)  # teal, BGR

    if rectified is not None:
        rect = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    else:
        rect = np.full((330, 330, 3), 30, np.uint8)
        cv2.putText(rect, "no grid", (110, 170), FONT, 0.7, (76, 87, 226), 2)

    if cells:
        cell = 330 // 3
        for i, cr in enumerate(cells):
            r, c = divmod(i, 3)
            x, y = c * cell, r * cell
            color = (163, 201, 53) if cr.mark else (110, 110, 110)
            if cr.mark and cr.conf < 0.75:
                color = (61, 163, 232)  # amber, BGR
            cv2.rectangle(rect, (x + 2, y + 2), (x + cell - 2, y + cell - 2), color, 1)
            label = f"{cr.mark or '-'} {cr.conf:.2f}"
            cv2.putText(rect, label, (x + 6, y + 18), FONT, 0.45, color, 1)
            cv2.putText(rect, f"ink {cr.ink_ratio:.2f}", (x + 6, y + cell - 8),
                        FONT, 0.38, (140, 140, 140), 1)

    canvas = np.full((view_h + BANNER_H, raw.shape[1] + 340, 3), 18, np.uint8)
    canvas[BANNER_H:, :raw.shape[1]] = raw
    canvas[BANNER_H:, raw.shape[1] + 10:raw.shape[1] + 340] = rect
    cv2.putText(canvas, banner, (8, 18), FONT, 0.5, (223, 230, 232), 1)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes() if ok else b""
