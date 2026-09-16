"""Session log + learned memory. Runtime data lives under runs/<GAME>/,
never inside the source packages — code is git, learned state is data.

events.jsonl: every FSM transition, perception summary per confirm, every
model call, game results. Decision records carry the provenance tuple (git
SHA, config hash, engine, resolved model id, thresholds, memory version) so
any move is reconstructable with one grep.

memory.json: per-setup learned state, loaded at boot, updated at game end,
inspectable by hand. Learned values apply only through the clamped ranges
hard-coded next to each threshold — a corrupted or drifted file cannot push
the system outside safe bounds.
"""

import json
import subprocess
import time
from pathlib import Path

from server.core import config as cfg

# hard bounds for anything memory is allowed to touch — the bad-update guard
CLAMPS = {
    "t_empty": (0.01, 0.08),
}

_DEFAULT_MEMORY = {
    "version": 1,
    "games_played": 0,
    "misreads": 0,
    "arbiter_calls": 0,
    "mismatches": 0,
    "t_empty": None,             # None = use the game config default
    "empty_ink_samples": [],     # rolling observations feeding t_empty
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cfg.ROOT,
            capture_output=True, text=True, timeout=2,
        ).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


class Store:
    def __init__(self, game_name: str, root: Path | None = None):
        self.dir = (root or cfg.RUNS_DIR) / game_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.memory_path = self.dir / "memory.json"
        self.memory = self._load_memory()
        self.provenance = {
            "git_sha": _git_sha(),
            "config_hash": cfg.config_hash(),
            "memory_version": self.memory["version"],
        }

    # --- events ---------------------------------------------------------------

    def log(self, kind: str, data: dict | None = None) -> None:
        record = {"ts": round(time.time(), 3), "type": kind, **(data or {})}
        with self.events_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_decision(self, kind: str, data: dict) -> None:
        """A decision record always carries the provenance tuple."""
        self.log(kind, {**data, "provenance": self.provenance})

    # --- memory ---------------------------------------------------------------

    def _load_memory(self) -> dict:
        if self.memory_path.exists():
            try:
                loaded = json.loads(self.memory_path.read_text())
                return {**_DEFAULT_MEMORY, **loaded}
            except (json.JSONDecodeError, TypeError):
                pass   # corrupted memory falls back to defaults, logged at boot
        return dict(_DEFAULT_MEMORY)

    def save_memory(self) -> None:
        self.memory["version"] += 1
        self.memory_path.write_text(json.dumps(self.memory, indent=2))
        self.provenance["memory_version"] = self.memory["version"]

    def learned_t_empty(self) -> float | None:
        """The only learned threshold today, always clamped on the way out."""
        v = self.memory.get("t_empty")
        if v is None:
            return None
        lo, hi = CLAMPS["t_empty"]
        return min(hi, max(lo, float(v)))

    def observe_empty_ink(self, samples: list[float]) -> None:
        buf = self.memory["empty_ink_samples"]
        buf.extend(round(s, 4) for s in samples)
        del buf[:-200]   # keep a rolling window

    def finish_game(self, result: str, moves: int, misreads: int,
                    arbiter_calls: int, mismatches: int) -> str:
        """Update memory at game end; returns the one learned line for the
        announcer. Update logic is deliberately trivial: t_empty moves toward
        (observed empty-cell ink p95 * 1.5), clamped."""
        m = self.memory
        m["games_played"] += 1
        m["misreads"] += misreads
        m["arbiter_calls"] += arbiter_calls
        m["mismatches"] += mismatches
        if len(m["empty_ink_samples"]) >= 30:
            samples = sorted(m["empty_ink_samples"])
            p95 = samples[int(len(samples) * 0.95)]
            lo, hi = CLAMPS["t_empty"]
            m["t_empty"] = round(min(hi, max(lo, p95 * 1.5)), 4)
        self.save_memory()
        self.log("game_result", {
            "result": result, "moves": moves, "misreads": misreads,
            "arbiter_calls": arbiter_calls, "mismatches": mismatches,
            "memory": {k: m[k] for k in
                       ("games_played", "misreads", "arbiter_calls", "t_empty")},
        })
        return (f"This session: {misreads} misreads, {arbiter_calls} arbiter "
                f"call{'s' if arbiter_calls != 1 else ''} — logged for next time. "
                f"Game {m['games_played']} on record.")
