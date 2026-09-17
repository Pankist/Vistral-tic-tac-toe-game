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


def test_calibration_majority_lock_tolerates_hiccups():
    """One dropped grid frame or one flickered reading must not reset the
    lock — real cameras hiccup every few frames."""
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    assert s.state == "CALIBRATING"
    feed(f, s, [""] * 9, P.n_calib - 2)
    f.step(s, frame([], grid=False))                    # dropped grid frame
    f.step(s, frame(["O"] + [""] * 8))                  # single flicker
    assert s.state == "CALIBRATING"
    effects = feed(f, s, [""] * 9, 2)                   # majority reached
    assert s.state == "HUMAN_TURN"
    assert points(effects) == ["lock"]


def test_prefilled_board_resumes_human_turn():
    """Lock on a page with a game in progress: marks become the board and
    equal counts hand the turn to the human."""
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    b = ["X", "", "", "", "O", "", "", "", ""]
    effects = feed(f, s, b, P.n_calib)
    assert s.state == "HUMAN_TURN"
    assert s.board == b
    assert len(s.history) == 2
    assert "lock_resume" in points(effects)


def test_prefilled_board_resumes_agent_turn():
    """x > o at lock: it's the agent's move — engine fires immediately."""
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    b = ["X", "", "", "", "O", "", "", "", "X"]
    effects = feed(f, s, b, P.n_calib)
    assert s.state == "AGENT_TURN"
    assert any(isinstance(e, EngineTurn) for e in effects)


def test_prefilled_finished_board_is_game_over():
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    b = ["X", "X", "X", "O", "O", "", "", "", ""]
    effects = feed(f, s, b, P.n_calib)
    assert s.state == "GAME_OVER"
    assert any(isinstance(e, GameEnded) and e.result == "human" for e in effects)


def test_calibration_needs_stable_readings_not_just_grid():
    """A grid that reads differently every frame must not lock — alternating
    hypotheses never reach the 70% majority."""
    f, s = fsm(), Session()
    f.step(s, ControlEvent("start"))
    for i in range(P.n_calib * 6):
        b = [""] * 9
        b[i % 2] = "X"                        # reading flaps between two cells
        f.step(s, frame(b))
    assert s.state == "CALIBRATING"


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


def test_fast_play_o_plus_next_x_confirms_both():
    """Human draws the asked-for O and their next X in one go — both moves
    confirm in order instead of freezing on a mismatch."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    assert s.state == "AWAIT_DRAW"
    both = list(s.board)
    both[4] = "O"                              # the O it asked for
    both[1] = "X"                              # plus the human's next move
    effects = feed(f, s, both, P.k_stable)
    assert s.board[4] == "O" and s.board[1] == "X"
    assert ("O", 4) in s.history and ("X", 1) in s.history
    assert s.state == "AGENT_TURN"
    assert any(isinstance(e, EngineTurn) for e in effects)
    assert "verified" in points(effects) and "react" in points(effects)


def test_await_mismatch_reports_the_offending_cell():
    """O landed where asked plus something illegal elsewhere: the challenge
    must name the offending cell, not the correct O."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    wrong = list(s.board)
    wrong[4] = "O"
    wrong[2] = "O"                             # second O — not a legal fast play
    effects = feed(f, s, wrong, P.k_stable)
    assert s.state == "MISMATCH"
    ann = [e for e in effects if isinstance(e, Announce) and e.point == "mismatch"]
    assert ann and ann[0].ctx["seen"] == "top-right"


def test_accept_reading_hands_turn_by_counts():
    """Accepted page has more X than O: the agent must move, not wait."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    seen = list(s.board)
    seen[2] = "O"                              # O in the wrong cell
    seen[5] = "X"                              # plus an extra human X
    feed(f, s, seen, P.k_stable)
    assert s.state == "MISMATCH"
    effects = f.step(s, ControlEvent("accept_reading"))
    assert s.state == "AGENT_TURN"             # x=2, o=1 → agent's move
    assert any(isinstance(e, EngineTurn) for e in effects)


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


def test_mismatch_self_heals_when_page_becomes_correct():
    """A transient misread (half-drawn O reads as X) trips MISMATCH; once the
    finished O reads correctly, the session must confirm and move on by
    itself — no button, no reset."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    wrong = list(s.board)
    wrong[4] = "X"                             # half-drawn O read as X
    effects = feed(f, s, wrong, P.k_stable)
    assert s.state == "MISMATCH"
    ann = [e for e in effects if isinstance(e, Announce)]
    assert any(a.point == "mismatch_wrong_mark" for a in ann)
    good = list(s.board)
    good[4] = "O"                              # the ink is finished now
    effects = feed(f, s, good, P.k_stable)
    assert s.state == "HUMAN_TURN"
    assert s.board[4] == "O"
    assert ("O", 4) in s.history
    assert "verified" in points(effects)


def test_mismatch_stays_quiet_on_identical_frames():
    """While frozen on the same wrong page, no re-announcing every frame."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("engine_move", cell=4))
    wrong = list(s.board)
    wrong[2] = "O"
    feed(f, s, wrong, P.k_stable)
    assert s.state == "MISMATCH"
    effects = feed(f, s, wrong, 10)            # same wrong page, 10 more frames
    assert points(effects) == []
    assert s.mismatches == 1


def test_human_turn_mismatch_heals_after_erasing_extra_mark():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X", "X"] + [""] * 7, P.k_stable)
    assert s.state == "MISMATCH"
    effects = feed(f, s, ["X"] + [""] * 8, P.k_stable)   # one X erased
    assert s.board[0] == "X"
    assert s.state == "AGENT_TURN"
    assert any(isinstance(e, EngineTurn) for e in effects)


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


def test_start_is_inert_mid_game():
    """Start begins watching; only Reset may abandon a game in progress."""
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    board_before = list(s.board)
    effects = f.step(s, ControlEvent("start"))
    assert s.board == board_before
    assert s.state != "CALIBRATING"
    assert points(effects) == []


def test_reset_mid_game():
    f, s = fsm(), Session()
    start_to_human_turn(f, s)
    feed(f, s, ["X"] + [""] * 8, P.k_stable)
    f.step(s, ControlEvent("reset"))
    assert s.state == "CALIBRATING"
    assert s.board == [""] * 9
    assert s.history == []
