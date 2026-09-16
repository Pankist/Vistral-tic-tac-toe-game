"""FSM on synthetic perception feeds — the full loop with no camera."""

import time

from server.core.fsm import (Announce, ArbiterCheck, ControlEvent, EngineTurn,
                             FSM, GameEnded, Params, Session)
from server.games.tic_tac_toe.game import TicTacToe
from server.perception.types import CellRead, PerceptionResult

P = Params(k_stable=4, n_calib=5, t_arbiter=0.75)


def fsm():
    return FSM(TicTacToe(), P)


def frame(labels, conf=0.95, motion=False, grid=True):
    return PerceptionResult(
        grid_found=grid, corners=None, reproj_error=0.1,
        cells=[CellRead(mark=m, conf=conf, ink_ratio=0.1 if m else 0.01)
               for m in labels] if grid else [],
        motion=motion, ts=time.time())


def feed(f, s, labels, n, conf=0.95):
    effects = []
    for _ in range(n):
        effects += f.step(s, frame(labels, conf))
    return effects


def start_to_human_turn(f, s):
    f.step(s, ControlEvent("start"))
    feed(f, s, [""] * 9, P.n_calib)
    assert s.state == "HUMAN_TURN"


def points(effects):
    return [e.point for e in effects if isinstance(e, Announce)]


def test_calibration_requires_consecutive_locks():
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    assert s.state == "CALIBRATING"
    feed(f, s, [""] * 9, P.n_calib - 1)
    assert s.state == "CALIBRATING"
    f.step(s, frame([], grid=False))          # streak broken
    feed(f, s, [""] * 9, P.n_calib - 1)
    assert s.state == "CALIBRATING"
    effects = feed(f, s, [""] * 9, 1)
    assert s.state == "HUMAN_TURN"
    assert points(effects) == ["lock"]


def test_debounce_confirms_after_k_stable_frames():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = ["X"] + [""] * 8
    feed(f, s, b, P.k_stable - 1)
    assert s.state == "CONFIRMING"
    assert s.board == [""] * 9                 # not confirmed yet
    effects = feed(f, s, b, 1)
    assert s.board[0] == "X"
    assert "react" in points(effects)
    assert any(isinstance(e, EngineTurn) for e in effects)
    assert s.state == "AGENT_TURN"


def test_label_flicker_resets_streak():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable - 1)
    feed(f, s, ["O"] + [""] * 8, 1)            # flicker: different hypothesis
    feed(f, s, ["X"] + [""] * 8, P.k_stable - 1)
    assert s.board == [""] * 9                 # streak restarted, still unconfirmed


def test_motion_frames_are_ignored_not_resetting():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = ["X"] + [""] * 8
    feed(f, s, b, P.k_stable - 1)
    f.step(s, frame(b, motion=True))           # hand in frame: gated
    feed(f, s, b, 1)
    assert s.board[0] == "X"


def test_full_game_all_announce_points():
    f, s = fsm(), Session()
    heard = []
    f_ = f

    def run(effects):
        heard.extend(points(effects))
        for e in effects:
            if isinstance(e, EngineTurn):
                # scripted agent replies for a deterministic X-win-free game
                run(f_.step(s, ControlEvent("engine_move", cell=agent_moves.pop(0))))

    # scripted to a known draw: X 0,8,3,2,7 / O 4,5,6,1 — no premature win
    agent_moves = [4, 5, 6, 1]
    run(f.step(s, ControlEvent("start")))
    heard.extend(points(feed(f, s, [""] * 9, P.n_calib)))
    b = [""] * 9
    for x_cell in [0, 8, 3, 2, 7]:
        b[x_cell] = "X"
        run(feed(f, s, b, P.k_stable))
        if s.state == "GAME_OVER":
            break
        assert s.state == "AWAIT_DRAW", f"after X {x_cell}: {s.state}"
        b[s.expected_cell] = "O"
        run(feed(f, s, b, P.k_stable))
    assert s.state == "GAME_OVER"
    assert s.result == "draw"
    for point in ("lock", "react", "own_move", "verified"):
        assert point in heard, f"missing announce point {point}"


def test_await_draw_wrong_cell_is_mismatch():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = ["X"] + [""] * 8
    feed(f, s, b, P.k_stable)
    assert s.state == "AGENT_TURN"
    f.step(s, ControlEvent("engine_move", cell=4))
    assert s.state == "AWAIT_DRAW" and s.expected_cell == 4
    wrong = list(s.board)
    wrong[2] = "O"                             # O landed top-right, not center
    effects = feed(f, s, wrong, P.k_stable)
    assert s.state == "MISMATCH"
    assert "mismatch" in points(effects)
    assert s.mismatch["expected_cell"] == 4


def test_mismatch_accept_reading_adopts_page():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    wrong = list(s.board)
    wrong[2] = "O"
    feed(f, s, wrong, P.k_stable)
    effects = f.step(s, ControlEvent("accept_reading"))
    assert "accepted" in points(effects)
    assert s.board[2] == "O"
    assert ("O", 2) in s.history               # the O counts where it landed
    assert s.state == "HUMAN_TURN"


def test_mismatch_fix_page_resumes_await():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    wrong = list(s.board)
    wrong[2] = "O"
    feed(f, s, wrong, P.k_stable)
    assert s.state == "MISMATCH"
    # human erases the stray O: page matches confirmed truth again
    effects = feed(f, s, s.board, P.k_stable)
    assert s.state == "AWAIT_DRAW"             # back to waiting for the center O
    assert s.expected_cell == 4
    assert "fixed" in points(effects)
    # then draws it right
    good = list(s.board)
    good[4] = "O"
    effects = feed(f, s, good, P.k_stable)
    assert s.board[4] == "O"
    assert "verified" in points(effects)


def test_illegal_two_cells_at_once_is_mismatch():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = ["X", "X"] + [""] * 7
    effects = feed(f, s, b, P.k_stable)
    assert s.state == "MISMATCH"
    assert "illegal" in points(effects)


def test_wrong_symbol_is_mismatch():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    effects = feed(f, s, ["O"] + [""] * 8, P.k_stable)
    assert s.state == "MISMATCH"
    assert "illegal" in points(effects)


def test_low_confidence_invokes_arbiter_then_resolves():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = ["X"] + [""] * 8
    effects = feed(f, s, b, P.k_stable, conf=0.5)
    assert s.arbiter_pending
    assert any(isinstance(e, ArbiterCheck) for e in effects)
    assert "arbiter_look" in points(effects)
    assert s.board == [""] * 9                 # decision parked
    payload = {"cells": [{"cell": i, "mark": m, "conf": 0.97}
                         for i, m in enumerate(b)]}
    effects = f.step(s, ControlEvent("arbiter_result", payload=payload))
    assert "arbiter_done" in points(effects)
    assert s.board[0] == "X"
    assert s.state == "AGENT_TURN"


def test_arbiter_failure_falls_back_to_human():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable, conf=0.5)
    effects = f.step(s, ControlEvent("arbiter_failed"))
    assert s.state == "MISMATCH"
    assert "arbiter_fail" in points(effects)


def test_human_win_ends_game():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    b = [""] * 9
    for cell in (0, 1):
        b[cell] = "X"
        feed(f, s, b, P.k_stable)
        assert s.state == "AGENT_TURN"
        f.step(s, ControlEvent("engine_move", cell=(cell + 3)))
        b[cell + 3] = "O"
        feed(f, s, b, P.k_stable)
    b[2] = "X"                                 # X completes the top row
    effects = feed(f, s, b, P.k_stable)
    assert s.state == "GAME_OVER"
    ends = [e for e in effects if isinstance(e, GameEnded)]
    assert ends and ends[0].result == "human"


def test_reset_mid_game():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("reset"))
    assert s.state == "CALIBRATING"
    assert s.board == [""] * 9
    assert s.history == []
