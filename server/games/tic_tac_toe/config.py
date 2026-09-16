"""Tic-tac-toe settings only — keys that exist nowhere else. A collision with
core config keys fails the boot assertion in core.config.merged().
Thresholds here are the [B] hot spots for mark reading; tune on-site.
"""

DEFAULT_ENGINE = "minimax"   # minimax | llm — the game is solved, so minimax is native
STRENGTH = "perfect"         # perfect | random — opponent-strength dial

# --- per-cell mark reading [B] --------------------------------------------------
T_EMPTY = 0.03               # ink_ratio below this = empty cell
CELL_MARGIN = 0.12           # inner margin cropped off each cell before reading
O_CIRCULARITY = 0.55         # outer contour circularity above this (with a hole) = O
MIN_HOLE_RATIO = 0.15        # child hole area / OUTER CONTOUR area — scale-free,
                             # so a small thin O is judged against itself, not the cell
X_CENTER_FRAC = 0.10         # ink fraction near the mark's own centroid ⇒ crossing ⇒ X
O_CENTER_FRAC = 0.04         # below this the center is empty ⇒ ring (even a broken one)

# --- escalation -----------------------------------------------------------------
T_ARBITER = 0.75             # aggregate confidence below this during a confirm → arbiter
