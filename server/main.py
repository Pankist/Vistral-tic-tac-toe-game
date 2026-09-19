"""FastAPI app: the single WS endpoint, /health, /config, static client.

One process, one session per WebSocket. The runner owns all I/O — perception
pipeline, engine, arbiter, announcer, store — and drives the pure FSM,
executing the effects it returns.
"""

import asyncio
import base64
import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from server.core import config as cfg
from server.core import fsm as fsm_mod
from server.core.announcer import Announcer
from server.core.arbiter import Arbiter, ArbiterUnavailable
from server.core.fsm import (Announce, ArbiterCheck, ControlEvent, EngineTurn,
                             FSM, GameEnded, LogEvent, Params, RecognizePuzzle,
                             Session, SolvePuzzle)
from server.core.llm import AnthropicClient
from server.core.loader import (load_engine, load_game, load_marks_module,
                                load_phrases, load_prompts_module)
from server.core.store import Store
from server.perception.pipeline import Pipeline
from server.perception.types import CellRead, PerceptionResult

app = FastAPI(title="visual-gamer-agent")
# the client is served from localhost while the backend sits behind the ALB —
# without CORS the /config fetch fails silently and the client runs on defaults
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

# effective config is always visible: printed at boot
_merged = cfg.merged()
print(f"[boot] config {cfg.config_hash()}: "
      + ", ".join(f"{k}={v}" for k, v in sorted(_merged.items())
                  if k not in ("ANTHROPIC_API_KEY",)))

_last_fsm_state = "IDLE"    # for /health


@app.get("/health")
def health():
    return {"status": "ok", "fsm": _last_fsm_state}


@app.get("/config")
def get_config():
    """Client fetches this on connect — one source of truth, no drift."""
    import importlib
    gcfg = importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.config")
    return {
        "fps": cfg.FPS, "jpeg_quality": cfg.JPEG_QUALITY,
        "k_stable": cfg.K_STABLE, "t_motion": cfg.T_MOTION,
        "game": cfg.ACTIVE_GAME, "engine": gcfg.DEFAULT_ENGINE,
        "announcer": cfg.ANNOUNCER, "rectify": cfg.RECTIFY,
        "voice": cfg.VOICE,
    }


@app.get("/")
def index():
    return FileResponse(cfg.ROOT / "client" / "index.html")


class SessionRunner:
    """Owns one game session's I/O; executes FSM effects."""

    def __init__(self, ws: WebSocket):
        import importlib
        self.ws = ws
        self.game = load_game()
        gcfg = importlib.import_module(f"server.games.{cfg.ACTIVE_GAME}.config")
        self.gcfg = gcfg
        self.store = Store(self.game.name)
        client = AnthropicClient()
        self.engine = load_engine(self.game, client, log=self.store.log)
        self.announcer = Announcer(load_phrases(), cfg.ANNOUNCER, client,
                                   log=self.store.log)
        self.arbiter = Arbiter(client, load_prompts_module(), log=self.store.log)
        self.fsm = FSM(self.game, Params(k_stable=cfg.K_STABLE,
                                         n_calib=cfg.N_CALIB,
                                         t_arbiter=gcfg.T_ARBITER))
        self.session = Session()
        self.pipeline = Pipeline(load_marks_module(), game_mode=cfg.ACTIVE_GAME)
        self.pipeline.t_empty_override = self.store.learned_t_empty()
        self.last_perception: PerceptionResult | None = None
        self._frame_times: list[float] = []
        self.message = "Press Start when the camera sees the page."

    # --- inbound --------------------------------------------------------------

    async def handle(self, msg: dict) -> None:
        if msg.get("type") == "frame":
            await self._on_frame(msg)
        elif msg.get("type") == "control":
            await self._on_control(msg)

    async def _on_frame(self, msg: dict) -> None:
        jpeg = base64.b64decode(msg["jpeg_b64"])
        banner = f"fsm {self.session.state}"
        result, dbg = await asyncio.to_thread(self.pipeline.process, jpeg, banner)
        self.last_perception = result
        self._frame_times = [t for t in self._frame_times if t > time.time() - 5]
        self._frame_times.append(time.time())
        self._collect_empty_ink(result)
        effects = self.fsm.step(self.session, result)
        await self._run_effects(effects)
        await self._send_state()
        if dbg:
            await self.ws.send_text(json.dumps(
                {"type": "debug", "jpeg_b64": base64.b64encode(dbg).decode()}))

    async def _on_control(self, msg: dict) -> None:
        action = msg.get("action", "")
        if action == "simulate_mark":
            await self._simulate(msg)
            return

        # Handle game switching
        if action == "switch_game":
            game_name = msg.get("game", "tic_tac_toe")
            await self._switch_game(game_name)
            return

        # Handle submarine-specific controls
        if action == "set_corners":
            corners = msg.get("corners", [])
            if len(corners) == 4:
                # Convert from client format to tuples
                self.pipeline.set_submarine_corners(
                    [(int(c[0]), int(c[1])) for c in corners]
                )

        effects = self.fsm.step(self.session, ControlEvent(
            action=action,
            cell=msg.get("cell"),
            mark=msg.get("mark"),
            payload=msg.get("corners") if action == "set_corners" else None))
        if action in ("start", "reset"):
            self.pipeline.hint = None
            self.pipeline.last_k = 0
        await self._run_effects(effects)
        await self._send_state()

    async def _switch_game(self, game_name: str) -> None:
        """Dynamically switch to a different game."""
        import importlib

        # Reload game components
        self.game = load_game(game_name)
        gcfg = importlib.import_module(f"server.games.{game_name}.config")
        self.gcfg = gcfg
        self.store = Store(self.game.name)
        client = AnthropicClient()
        self.engine = load_engine(self.game, client, log=self.store.log, game_name=game_name)
        self.announcer = Announcer(load_phrases(game_name), cfg.ANNOUNCER, client,
                                   log=self.store.log)
        self.arbiter = Arbiter(client, load_prompts_module(game_name), log=self.store.log)

        # Reload pipeline with new marks module
        marks = load_marks_module(game_name)
        if marks:
            self.pipeline = Pipeline(marks, game_mode=game_name)
        else:
            # Submarine doesn't have marks
            self.pipeline = Pipeline(None, game_mode=game_name)

        # Recreate FSM with new game
        self.fsm = FSM(self.game, Params(k_stable=cfg.K_STABLE,
                                         n_calib=cfg.N_CALIB,
                                         t_arbiter=getattr(gcfg, 'T_ARBITER', 0.75)))

        # Reset session
        self.session = Session()
        self.pipeline.hint = None
        self.pipeline.last_k = 0
        self.message = "Game switched. Press Start when ready."

        await self._send_state()

    async def _simulate(self, msg: dict) -> None:
        """Dev path (?dev=1): inject synthetic stable PerceptionResults so the
        full loop — calibration, debounce, confirm — runs with no camera."""
        s = self.session
        if s.state == "IDLE":
            await self._on_control({"action": "start"})
        if s.state == "CALIBRATING":
            await self._feed_synthetic(s.board, cfg.N_CALIB)
        cell = int(msg.get("cell", -1))
        if cell < 0 or s.state not in ("HUMAN_TURN", "CONFIRMING", "AWAIT_DRAW",
                                       "MISMATCH"):
            await self._send_state()
            return
        mark = msg.get("mark") or (self.game.agent_mark
                                   if s.state == "AWAIT_DRAW"
                                   else self.game.human_mark)
        board = list(s.board)
        board[cell] = mark
        await self._feed_synthetic(board, cfg.K_STABLE)
        await self._send_state()

    async def _feed_synthetic(self, labels: list[str], n: int) -> None:
        for _ in range(n):
            p = PerceptionResult(
                grid_found=True, corners=None, reproj_error=0.0,
                cells=[CellRead(mark=m, conf=0.99, ink_ratio=0.1 if m else 0.01)
                       for m in labels],
                motion=False, ts=time.time(), method="sim")
            self.last_perception = p
            await self._run_effects(self.fsm.step(self.session, p))

    # --- effects --------------------------------------------------------------

    async def _run_effects(self, effects: list) -> None:
        global _last_fsm_state
        for eff in effects:
            if isinstance(eff, Announce):
                text = self.announcer.render(eff.point, **eff.ctx)
                self.message = text
                self.store.log("speak", {"point": eff.point, "text": text})
                await self.ws.send_text(json.dumps({"type": "speak", "text": text}))
            elif isinstance(eff, EngineTurn):
                await self._engine_turn()
            elif isinstance(eff, ArbiterCheck):
                asyncio.get_running_loop().create_task(self._arbiter_turn(eff))
            elif isinstance(eff, RecognizePuzzle):
                asyncio.get_running_loop().create_task(self._recognize_puzzle(eff))
            elif isinstance(eff, SolvePuzzle):
                asyncio.get_running_loop().create_task(self._solve_puzzle(eff))
            elif isinstance(eff, GameEnded):
                await self._game_over(eff)
            elif isinstance(eff, LogEvent):
                if eff.kind == "confirm":
                    self.pipeline.hint = list(self.session.board)
                    self.store.log_decision(eff.kind, eff.data)
                else:
                    if eff.kind == "board_lock":
                        # prefilled marks anchor orientation from the start
                        self.pipeline.hint = list(self.session.board)
                    self.store.log(eff.kind, eff.data)
        _last_fsm_state = self.session.state

    async def _engine_turn(self) -> None:
        board, history = list(self.session.board), list(self.session.history)
        t0 = time.perf_counter()
        move = await asyncio.to_thread(self.engine.decide, board, history)
        ms = (time.perf_counter() - t0) * 1000
        # the FSM validates every move regardless of engine
        if move not in self.game.legal_moves(board):
            self.store.log("engine_illegal", {"engine": self.engine.name,
                                              "move": move})
            move = self.game.legal_moves(board)[0]
        self.store.log_decision("engine_move", {
            "engine": self.engine.name, "move": move, "ms": round(ms, 2),
            "reasoning": getattr(self.engine, "last_reasoning", "")})
        await self._run_effects(self.fsm.step(
            self.session, ControlEvent(action="engine_move", cell=move)))

    async def _arbiter_turn(self, eff: ArbiterCheck) -> None:
        rect = self.pipeline.last_rectified
        try:
            if rect is None:
                raise ArbiterUnavailable("no rectified frame available")
            parsed = await asyncio.to_thread(
                self.arbiter.read_board, rect, self.session.board, eff.detail)
            event = ControlEvent(action="arbiter_result", payload=parsed)
        except ArbiterUnavailable as e:
            self.store.log("arbiter_failed", {"error": str(e)})
            event = ControlEvent(action="arbiter_failed")
        await self._run_effects(self.fsm.step(self.session, event))
        await self._send_state()

    async def _recognize_puzzle(self, eff: RecognizePuzzle) -> None:
        """Submarine: recognize puzzle from image using vision."""
        if cfg.ACTIVE_GAME != "submarine":
            return

        from server.games.submarine.engine import SubmarineEngine
        engine = SubmarineEngine(log=self.store.log)

        rect = self.pipeline.last_rectified or eff.image
        try:
            if rect is None:
                raise Exception("no image available")
            result = await asyncio.to_thread(engine.recognize_puzzle, rect)
            event = ControlEvent(action="recognize_result", payload=result)
        except Exception as e:
            self.store.log("recognition_failed", {"error": str(e)})
            event = ControlEvent(action="recognize_result",
                                payload={"has_puzzle": False})
        await self._run_effects(self.fsm.step(self.session, event))
        await self._send_state()

    async def _solve_puzzle(self, eff: SolvePuzzle) -> None:
        """Submarine: solve the recognized puzzle."""
        if cfg.ACTIVE_GAME != "submarine":
            return

        from server.games.submarine.engine import SubmarineEngine
        engine = SubmarineEngine(log=self.store.log)

        try:
            solution = await asyncio.to_thread(
                engine.solve_puzzle, eff.puzzle_text, eff.options)
            event = ControlEvent(action="solve_result", payload=solution)
        except Exception as e:
            self.store.log("solving_failed", {"error": str(e)})
            event = ControlEvent(action="solve_result",
                                payload={"answer": "Error", "reasoning": str(e)})
        await self._run_effects(self.fsm.step(self.session, event))
        await self._send_state()

    async def _game_over(self, eff: GameEnded) -> None:
        s = self.session
        result_text = {"human": "You win", "agent": "I win",
                       "draw": "Draw"}[eff.result]
        summary = self.store.finish_game(
            result=eff.result, moves=len(s.history), misreads=s.misreads,
            arbiter_calls=s.arbiter_calls, mismatches=s.mismatches)
        self.pipeline.t_empty_override = self.store.learned_t_empty()
        text = self.announcer.render("game_over", result=result_text,
                                     summary=summary)
        self.message = text
        self.store.log("speak", {"point": "game_over", "text": text})
        await self.ws.send_text(json.dumps({"type": "speak", "text": text}))

    def _collect_empty_ink(self, p: PerceptionResult) -> None:
        if p.grid_found and self.session.state in ("HUMAN_TURN", "AWAIT_DRAW"):
            samples = [c.ink_ratio for i, c in enumerate(p.cells)
                       if self.session.board[i] == "" and c.mark == ""]
            if samples:
                self.store.observe_empty_ink(samples)

    # --- outbound -------------------------------------------------------------

    async def _send_state(self) -> None:
        s, p = self.session, self.last_perception
        x = sum(1 for m in s.board if m == self.game.human_mark)
        o = sum(1 for m in s.board if m == self.game.agent_mark)
        turn = ("human" if s.state in ("HUMAN_TURN", "CONFIRMING")
                else "agent" if s.state in ("AGENT_TURN", "AWAIT_DRAW")
                else "-")

        # Base payload
        payload = {
            "type": "state", "fsm": s.state, "board": s.board, "turn": turn,
            "confidence": s.last_conf, "message": self.message,
            "expected_cell": s.expected_cell,
            "mismatch": s.mismatch, "move_no": x + o,
            "win_line": (self.game.win_line(s.board)
                         if s.state == "GAME_OVER" else None),
            "telemetry": {
                "fps": round(len(self._frame_times) / 5.0, 1),
                "diff": p.diff if p else 0.0, "t_motion": cfg.T_MOTION,
                "motion": p.motion if p else False,
                "grid_found": p.grid_found if p else False,
                "method": p.method if p else "",
                "reproj": p.reproj_error if p else None,
                "corners": p.corners if p else None,
                "cells": [{"mark": c.mark, "conf": c.conf, "ink": c.ink_ratio}
                          for c in (p.cells if p else [])],
                "streak": s.streak, "k_stable": cfg.K_STABLE,
                "arbiter": "pending" if s.arbiter_pending else "idle",
                "frame_ms": round(self.pipeline.last_ms, 1),
                "engine": self.engine.name,
            },
        }

        # Add submarine-specific fields
        if self.game.name == "submarine":
            payload["submarine"] = {
                "puzzle_text": s.puzzle_text,
                "puzzle_options": s.puzzle_options,
                "answer": s.answer,
                "settle_count": s.settle_count,
                "change_info": p.change_info if p and hasattr(p, 'change_info') else None,
            }

        await self.ws.send_text(json.dumps(payload))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    runner = SessionRunner(ws)
    runner.store.log("session_start", runner.store.provenance)
    await runner._send_state()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await runner.handle(msg)
    except WebSocketDisconnect:
        runner.store.log("session_end", {"fsm": runner.session.state})
    except Exception as e:                      # log, never crash the process
        runner.store.log("session_error", {"error": repr(e)})
        raise
