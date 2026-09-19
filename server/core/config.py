"""Core config — system mechanics, game-blind.

Three disjoint sources (no overrides, no precedence chain):
  1. this file          — mechanics and system choices, git-tracked, canonical
  2. games/<G>/config.py — game settings, keys that exist nowhere else
  3. .env                — secrets and deployment facts only, never behavioral

A key collision between (1) and (2) is a boot-time assertion failure — see
`merged()`. Thresholds marked [B] are the empirically-tuned hot spots: expect
to edit them on-site; `make dev` hot-reloads a save in ~2s.
"""

import hashlib
import json
import os
from pathlib import Path

# --- which game is mounted ---------------------------------------------------
ACTIVE_GAME = os.environ.get("ACTIVE_GAME", "tic_tac_toe")  # tic_tac_toe | submarine
AVAILABLE_GAMES = ["tic_tac_toe", "submarine"]

# --- frame ingestion ----------------------------------------------------------
FPS = 4                      # client obeys the server: fetched via GET /config
JPEG_QUALITY = 0.7
FRAME_WIDTH = 640            # downscale target before any CV

# --- perception mechanics [B] ---------------------------------------------------
T_MOTION = 6.0               # mean abs diff vs previous frame; above = hand in frame
K_STABLE = 4                 # identical symbolic hypotheses required to confirm
N_CALIB = 5                  # consecutive grid locks required to leave CALIBRATING
RECTIFY = "auto"             # lines | aruco | auto (auto = lines; markers if present)
CANONICAL = 330              # rectified board is CANONICAL x CANONICAL px

# --- speech -------------------------------------------------------------------
ANNOUNCER = "template"       # template | llm (llm adds one flavor line, 2s budget)
VOICE = "on"                 # on | off — client speechSynthesis default (UI can toggle)

# --- models (used only when confidence drops / engine=llm) ---------------------
MODEL_ARBITER = "claude-sonnet-5"
MODEL_ENGINE = "claude-sonnet-5"
MODEL_ANNOUNCER = "claude-haiku-4-5-20251001"   # flavor lines: speed over depth
LLM_TIMEOUT_S = 6.0
ANNOUNCER_BUDGET_S = 2.0

# --- paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"

# --- .env (secrets / deployment facts only) ------------------------------------

def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))
S3_BUCKET = os.environ.get("S3_BUCKET", "")

# env may override tunables for on-site rescue without touching git state
T_MOTION = float(os.environ.get("T_MOTION", T_MOTION))


def _public(mod) -> dict:
    return {
        k: v for k, v in vars(mod).items()
        if k.isupper() and not k.startswith("_")
    }


_SECRET_KEYS = {"ANTHROPIC_API_KEY"}


def merged() -> dict:
    """Core + active game's config in one namespace. Collision = boot failure."""
    import importlib
    import sys
    game_cfg = importlib.import_module(f"server.games.{ACTIVE_GAME}.config")
    core = _public(sys.modules[__name__])
    game = _public(game_cfg)
    collisions = set(core) & set(game)
    assert not collisions, f"config key collision between core and game: {collisions}"
    out = {**core, **game}
    out.pop("ROOT", None)
    out.pop("RUNS_DIR", None)
    return out


def config_hash() -> str:
    """Stable hash of every effective (non-secret) config value — provenance."""
    cfg = {k: v for k, v in merged().items() if k not in _SECRET_KEYS}
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
