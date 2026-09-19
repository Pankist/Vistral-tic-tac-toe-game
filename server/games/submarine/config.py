"""Submarine game settings — puzzle recognition and solving parameters."""

# Engine (submarine uses vision-based solver, not minimax)
DEFAULT_ENGINE = "submarine_solver"

# How many consecutive stable frames before triggering recognition
SETTLE_FRAMES = 8

# Frame difference threshold for change detection (0-1, lower = more sensitive)
CHANGE_THRESHOLD = 0.05

# Vision model for puzzle recognition
MODEL_RECOGNIZER = "claude-fable-5-1"

# Model for solving puzzles
MODEL_SOLVER = "claude-sonnet-5"

# Timeout for vision calls
VISION_TIMEOUT_S = 10.0

# Arbiter threshold (not used in submarine, but required by config merge)
T_ARBITER = 0.75
