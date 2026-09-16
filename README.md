# visual-gamer-agent

An AI agent that plays board games against a human over live video. The human draws on paper; the agent watches through a camera, detects new marks unprompted, decides its move, announces it out loud, and verifies the human actually drew it before play continues. Ink on the page is the source of truth — a mark the agent hasn't seen doesn't exist.

First game: tic-tac-toe (`games/tic_tac_toe/`). The core is game-agnostic — perception handles geometry, the FSM owns the session, and engines plug in behind a protocol. `games/chess/` is empty on purpose: that's where the next game goes.

**The agent is the loop, not the LLM.** Continuous perception is classical CV (OpenCV, ~free, ≤40ms/frame). A vision model is consulted only when confidence drops; move selection is minimax by default, with a swappable LLM engine (one config line) guarded by the same legality validation. A clean game costs $0.00 in inference.

## Quick start (under 10 minutes)

Prereqs: Python 3.12, a webcam (laptop, USB, or phone-as-webcam), a sheet of paper, a thick dark marker.

```bash
git clone https://github.com/Pankist/Vistral-tic-tac-toe-game.git && cd Vistral-tic-tac-toe-game
make setup                        # venv + pinned deps
cp .env.example .env              # add OPENROUTER_API_KEY (optional — see below)
make print-board                  # boards.pdf: optional tidy printable grid
make dev                          # server + client on http://localhost:8000
```

Open `http://localhost:8000`, allow camera access, draw a bold `#` grid on the page (thick strokes, no border needed), point the camera down at it, press **Start**.

No API key? Everything runs except the low-confidence arbiter and the optional LLM engine — the game plays fine without them.

Dry run without a camera: open `http://localhost:8000?dev=1` and use the simulate-mark grid to play a full game against the engine.

## How to play

1. **Calibration.** Hold the camera steady over the page. When the agent locks the grid it tells you so, with its confidence. The debug pane (toggle in the UI) shows what it sees: raw frame, rectified board, per-cell reads. A page with a game already on it works too: the marks become the starting position, and mark counts decide whose turn it is ("2 X and 1 O already played — my move").
2. **Your move.** Draw an X in any cell, then take your hand out of frame. The agent ignores frames while your hand is moving and accepts a mark only after it's been stable for several frames — expect ~1–2s from pen-up to acknowledgment.
3. **Its move.** The agent announces its cell on screen and (voice on) out loud. **You draw its O** in that cell. It watches, confirms the mark landed where it asked, and hands the turn back. Drawing its O and your next X in one go is fine — it confirms both, in order.
4. **Disagreements.** If the page doesn't match what the agent expects — wrong cell, ambiguous mark — it says so and asks: fix the page, or tell it to accept the reading. Fixing the ink is enough on its own: the agent re-judges the corrected page and play continues without a click. If it's unsure about a mark, it takes a closer look (one vision-model call) before deciding.
5. Play to win or draw. At game end it reports the result and what it logged for next time.

Tips: even, non-glare lighting; keep the page flat and the camera fixed; draw thick marks that fill most of the cell.

## Configuration

Behavior lives in two git-tracked config files; `.env` holds only secrets and deployment facts (`OPENROUTER_API_KEY`, port, optional `S3_BUCKET`). Change anything by editing the file — `make dev` hot-reloads it in ~2s.

| Where | Keys | Meaning |
|---|---|---|
| `server/core/config.py` | `ACTIVE_GAME`, `FPS`, `T_MOTION`, `K_STABLE`, `RECTIFY`, `ANNOUNCER`, `VOICE`, model names | System mechanics: which game is mounted, how frames are read and gated, rectification mode, how the agent speaks (`VOICE="off"` boots the client silent; the UI can toggle per session) |
| `server/games/tic_tac_toe/config.py` | `DEFAULT_ENGINE`, `STRENGTH`, `T_empty`, classification cutoffs, `T_arbiter` | Game settings: engine choice, opponent strength, perception thresholds for this game |

## Architecture in one paragraph

Browser client (camera capture, speech, debug view) streams JPEG frames at ~4 FPS over WebSocket to a single FastAPI process. Perception (motion gate → rectify → per-cell read) produces a `PerceptionResult`; the FSM — a pure function over states `CALIBRATING → HUMAN_TURN → CONFIRMING → AGENT_TURN → AWAIT_DRAW → GAME_OVER`, plus `MISMATCH` — turns stable readings into game events; the configured engine picks the reply; the announcer speaks it. Every transition, perception summary, and model call lands in `runs/events.jsonl`; per-setup calibration accumulates in `runs/memory.json` (inspectable JSON, survives restarts). Full design, including the game-agnostic expansion: `docs/architecture.md`.

```
client (localhost) ──ws frames──▶ FastAPI: perception → FSM → engine → announcer ──state/speech──▶ client
                                              └── low confidence only ──▶ vision arbiter (OpenRouter)
```

## Decisions, briefly

- **Minimax over LLM for moves** — deterministic, verifiable, 0ms, $0. The LLM engine exists behind the same interface (one config line) and is game-agnostic; legality validation in the FSM contains it either way.
- **Expensive inference at the edge** — the model sees one rectified image, only when classical CV admits uncertainty. Zero model calls in a clean game.
- **Markerless grid detection** — the one shipping path reads a plain hand-drawn `#` via Hough line families and their intersections, rotation and drift included. A fiducial (ArUco) fallback is deliberately *not* implemented; `perception/rectify_aruco.py` is a documented placeholder marking where it would go if a venue's lighting ever demanded it.
- **Debounce + motion gate over per-frame reads** — a mark exists when it survives K still frames, which is what makes half-drawn marks and hands-in-frame non-events.
- **Agent verifies its own moves** — the human draws for it, so `AWAIT_DRAW` confirms the ink matches the announcement before play continues.
- **No agent framework, no DB, one process** — the loop is a state machine; JSONL and a JSON file are enough state, and everything stays inspectable.

## Deploy (EC2, optional)

The demo runs local-first. The same repo deploys unchanged:

```bash
git clone https://github.com/Pankist/Vistral-tic-tac-toe-game.git app && cd app && make setup
sudo cp deploy/ttt-agent.service /etc/systemd/system/ && sudo systemctl enable --now ttt-agent
```

The exact AWS topology used for the live demo (dedicated VPC, EC2, ALB) is recorded in `deploy/aws.md`.

Behind an ALB: target group → :8000, health check `GET /health`, **idle timeout 600s** (the default 60s kills long-lived WebSockets). The client ships with the deployed ALB as its default backend (`DEPLOYED_BACKEND` in `client/index.html`); `?backend=ws://<host>/ws` overrides it, and a page served by the FastAPI app itself talks to its own host. Keep the client local — camera capture requires a secure context, which localhost is. Serve it with `make client` (`CLIENT_PORT=3001` if 3000 is taken).

## Troubleshooting

- **No camera prompt** — you're not on localhost / a secure context; serve the client locally.
- **Grid won't lock** — fix glare, flatten the page, redraw the `#` with thicker strokes; watch the reprojection number in the debug pane.
- **Marks not registering** — draw thicker, fill more of the cell, keep your hand fully out of frame; watch the per-cell reads in the debug pane. A thick marker beats a ballpoint every time, though thin pens do work.
- **Wrong lock on a dark desk** — the detector only accepts lines that are ink-dark with equally bright paper on both sides, which rejects paper edges and shadows; if a lock still looks wrong, check for glare and stray doodles near the grid (ink outside the grid but within a cell's reach is read as a mark — that's the stated assumption).
- **Everything looks right but it's wrong** — read `runs/events.jsonl` for the last confirm: it records exactly what perception saw and why the FSM did what it did.

## Repo map

```
server/core/        game/engine protocols, FSM, store, LLM plumbing — game-blind
server/perception/  geometry only: motion gate, rectification, debug composite
server/games/       one package per game; tic_tac_toe implemented, chess/ marks the seam
client/             one HTML file: camera, speech, debug view
scripts/            board PDF generator, arbiter smoke test
tests/              game, engines, FSM on synthetic perception feeds
```

Demo video: `docs/demo.mp4` <!-- record after rehearsal -->
