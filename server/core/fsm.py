"""The session brain: states CALIBRATING -> HUMAN_TURN -> CONFIRMING ->
AGENT_TURN -> AWAIT_DRAW -> GAME_OVER, plus MISMATCH (and IDLE before Start).

`step(session, event)` is deterministic and I/O-free: it consumes a
PerceptionResult or a ControlEvent and returns effects — announcements,
engine/arbiter requests, log records — for the runtime to execute. That makes
the whole game loop unit-testable with synthetic perception feeds.

Debounce lives here, not in perception: a hypothesis is the 9 symbolic
labels; two frames match iff all labels match, regardless of confidence
wobble. A borderline cell flickering its label correctly resets the streak —
that reading isn't evidence yet.
"""

from dataclasses import dataclass, field
from typing import Any

from server.perception.types import PerceptionResult

# --- events -------------------------------------------------------------------


@dataclass
class ControlEvent:
    action: str                     # start | reset | engine_move | arbiter_result |
                                    # arbiter_failed | accept_reading | fix_page
    cell: int | None = None
    mark: str | None = None
    payload: Any = None


# --- effects ------------------------------------------------------------------


@dataclass
class Announce:
    point: str
    ctx: dict = field(default_factory=dict)


@dataclass
class EngineTurn:
    pass


@dataclass
class ArbiterCheck:
    cell: int
    detail: str


@dataclass
class GameEnded:
    result: str                     # "human" | "agent" | "draw"


@dataclass
class RecognizePuzzle:
    """Submarine: trigger vision-based puzzle recognition."""
    image: Any = None


@dataclass
class SolvePuzzle:
    """Submarine: trigger puzzle solving."""
    puzzle_text: str = ""
    options: list[str] = field(default_factory=list)


@dataclass
class LogEvent:
    kind: str
    data: dict


@dataclass
class Params:
    k_stable: int = 4
    n_calib: int = 5
    t_arbiter: float = 0.75


@dataclass
class Session:
    state: str = "IDLE"
    board: list[str] = field(default_factory=lambda: [""] * 9)
    history: list[tuple[str, int]] = field(default_factory=list)
    calib_window: list = field(default_factory=list)
    candidate: tuple | None = None
    streak: int = 0
    expected_cell: int | None = None
    mismatch: dict | None = None
    arbiter_pending: bool = False
    last_conf: float = 0.0
    result: str | None = None
    # per-game counters, persisted by the runner at game end
    misreads: int = 0
    arbiter_calls: int = 0
    mismatches: int = 0

    # submarine-specific state
    puzzle_text: str = ""
    puzzle_options: list[str] = field(default_factory=list)
    answer: str = ""
    corners: list[tuple[int, int]] = field(default_factory=list)
    settle_count: int = 0

    def reset_candidate(self) -> None:
        self.candidate, self.streak = None, 0


class FSM:
    def __init__(self, game, params: Params | None = None):
        self.game = game
        self.p = params or Params()

    # --- entry point ----------------------------------------------------------

    def step(self, s: Session, event) -> list:
        if isinstance(event, ControlEvent):
            return self._control(s, event)
        if isinstance(event, PerceptionResult):
            return self._perception(s, event)
        return []

    # --- control events -------------------------------------------------------

    def _control(self, s: Session, e: ControlEvent) -> list:
        # Submarine-specific controls
        if e.action == "set_corners":
            s.corners = e.payload or []
            return [LogEvent("control", {"action": "set_corners", "corners": s.corners})]

        if e.action == "rescan":
            s.state = "MONITORING"
            s.settle_count = 0
            s.puzzle_text = ""
            s.answer = ""
            return [Announce("monitoring"), LogEvent("control", {"action": "rescan"})]

        if e.action == "recognize_result":
            result = e.payload or {}
            if result.get("has_puzzle"):
                s.puzzle_text = result.get("puzzle_text", "")
                s.puzzle_options = result.get("options", [])
                s.state = "SOLVING"
                return [
                    SolvePuzzle(s.puzzle_text, s.puzzle_options),
                    Announce("solving", option_count=len(s.puzzle_options)),
                    LogEvent("puzzle_recognized", {"puzzle": s.puzzle_text})
                ]
            else:
                s.state = "MONITORING"
                s.settle_count = 0
                return [Announce("no_puzzle"), LogEvent("no_puzzle_found", {})]

        if e.action == "solve_result":
            solution = e.payload or {}
            s.answer = solution.get("answer", "Error")
            s.state = "STANDBY"
            return [
                Announce("solved", {"answer": s.answer}),
                LogEvent("puzzle_solved", {"answer": s.answer, "reasoning": solution.get("reasoning")})
            ]

        if e.action in ("start", "reset"):
            # Start begins watching and is inert mid-game (no accidental
            # wipes); Reset is the deliberate abort and works anywhere.
            if e.action == "start" and s.state not in ("IDLE", "GAME_OVER"):
                return [LogEvent("control", {"action": "start",
                                             "ignored_in": s.state})]
            # For submarine, start goes to MONITORING; for tic_tac_toe, CALIBRATING
            initial_state = "MONITORING" if self.game.name == "submarine" else "CALIBRATING"
            fresh = Session(state=initial_state)
            s.__dict__.update(fresh.__dict__)
            point = "start" if e.action == "start" else "reset"
            return [Announce(point), LogEvent("control", {"action": e.action})]

        if e.action == "engine_move" and s.state == "AGENT_TURN":
            s.expected_cell = e.cell
            s.state = "AWAIT_DRAW"
            s.reset_candidate()
            return [Announce("own_move", {"cell": self.game.describe_cell(e.cell)}),
                    LogEvent("agent_move_announced", {"cell": e.cell})]

        if e.action == "arbiter_result" and s.arbiter_pending:
            s.arbiter_pending = False
            labels = tuple(c["mark"] for c in sorted(e.payload["cells"],
                                                     key=lambda c: c["cell"]))
            confs = [float(c["conf"]) for c in e.payload["cells"]]
            return self._resolve_stable(s, labels, min(confs), via_arbiter=True)

        if e.action == "arbiter_failed" and s.arbiter_pending:
            s.arbiter_pending = False
            self._enter_mismatch(s, seen=list(s.candidate or s.board),
                                 detail="ambiguous mark, arbiter unavailable")
            return [Announce("arbiter_fail"),
                    LogEvent("mismatch", {"detail": "arbiter unavailable"})]

        if e.action == "accept_reading" and s.state == "MISMATCH":
            return self._accept_reading(s)

        if e.action == "fix_page" and s.state == "MISMATCH":
            return [Announce("fixing")]

        return []

    # --- perception events ----------------------------------------------------

    def _perception(self, s: Session, p: PerceptionResult) -> list:
        # Submarine game perception flow
        if self.game.name == "submarine":
            if s.state == "MONITORING":
                # Check if image has settled (no changes for N frames)
                change_info = p.change_info if hasattr(p, 'change_info') else None
                if change_info:
                    if change_info.get('settled'):
                        s.state = "RECOGNIZING"
                        s.settle_count = change_info.get('stable_count', 0)
                        return [
                            RecognizePuzzle(image=p.raw_frame if hasattr(p, 'raw_frame') else None),
                            Announce("settled"),
                            LogEvent("image_settled", {"settle_count": s.settle_count})
                        ]
                return []
            # Other submarine states don't process perception
            return []

        # Tic-tac-toe perception flow
        if p.motion or s.state in ("IDLE", "AGENT_TURN", "GAME_OVER"):
            return []

        if s.state == "CALIBRATING":
            if p.grid_found:
                # sliding-window majority, not a consecutive streak: one
                # flickering borderline cell or one dropped grid frame must
                # not reset the lock — but a genuinely flapping reading never
                # reaches majority and never becomes the starting truth
                labels = p.labels()
                s.calib_window.append(labels)
                del s.calib_window[:-10]
                n = s.calib_window.count(labels)
                if n >= self.p.n_calib and n / len(s.calib_window) >= 0.7:
                    s.board = list(labels)
                    s.calib_window = []
                    s.last_conf = _mean_conf(p)
                    return self._lock(s)
            return []

        if not p.grid_found:
            return []

        if s.state in ("HUMAN_TURN", "CONFIRMING", "AWAIT_DRAW", "MISMATCH"):
            return self._watch(s, p)
        return []

    def _lock(self, s: Session) -> list:
        """Board lock — empty page starts fresh; a prefilled position resumes:
        the marks on the page become the confirmed board and mark counts say
        whose turn it is. Ink is the source of truth, including old ink."""
        effects = [LogEvent("board_lock", {"conf": s.last_conf, "board": s.board})]
        x = sum(1 for m in s.board if m == self.game.human_mark)
        o = sum(1 for m in s.board if m == self.game.agent_mark)
        s.history = [(m, i) for i, m in enumerate(s.board) if m != ""]
        if x == 0 and o == 0:
            s.state = "HUMAN_TURN"
            return [Announce("lock", {"conf": int(s.last_conf * 100)})] + effects
        if self.game.winner(s.board) or self.game.is_draw(s.board):
            s.state = "GAME_OVER"
            w = self.game.winner(s.board)
            s.result = ("draw" if not w
                        else "human" if w == self.game.human_mark else "agent")
            return effects + [GameEnded(result=s.result)]
        resume_ctx = {"conf": int(s.last_conf * 100), "xs": x, "os": o}
        if x > o:
            s.state = "AGENT_TURN"
            return [Announce("lock_resume", {**resume_ctx,
                                             "turn_line": "My move — one second."}),
                    EngineTurn()] + effects
        s.state = "HUMAN_TURN"
        return [Announce("lock_resume", {**resume_ctx,
                                         "turn_line": "Your move."})] + effects

    def _watch(self, s: Session, p: PerceptionResult) -> list:
        labels = p.labels()
        if list(labels) == s.board:
            # page matches confirmed truth
            if s.state == "CONFIRMING":
                s.state = "HUMAN_TURN"          # candidate evaporated
            if s.state == "MISMATCH":
                return self._resume_from_mismatch(s)
            s.reset_candidate()
            return []

        if labels == s.candidate:
            s.streak += 1
        else:
            s.candidate, s.streak = labels, 1
            if s.state == "HUMAN_TURN":
                s.state = "CONFIRMING"
        if s.streak < self.p.k_stable:
            return []

        # stable: changed-cell confidence gates the decision
        changed = _diff(s.board, labels)
        conf = min((p.cells[c].conf for c, _, _ in changed), default=1.0)
        if conf < self.p.t_arbiter and not s.arbiter_pending and s.state != "MISMATCH":
            s.arbiter_pending = True
            s.arbiter_calls += 1
            cell = changed[0][0] if changed else 0
            return [Announce("arbiter_look", {"cell": self.game.describe_cell(cell)}),
                    ArbiterCheck(cell=cell, detail=str(changed)),
                    LogEvent("arbiter_invoked", {"changed": changed, "conf": conf})]
        if s.arbiter_pending:
            return []          # decision parked until the arbiter reports
        return self._resolve_stable(s, labels, conf)

    # --- the confirm decision -------------------------------------------------

    def _resolve_stable(self, s: Session, labels: tuple, conf: float,
                        via_arbiter: bool = False) -> list:
        effects: list = []
        changed = _diff(s.board, labels)
        s.reset_candidate()
        if not changed:
            if s.state == "MISMATCH":
                return self._resume_from_mismatch(s)
            if s.state == "CONFIRMING":
                s.state = "HUMAN_TURN"
            return []
        if via_arbiter:
            cell, _, new = changed[0]
            effects.append(Announce("arbiter_done", {
                "cell": self.game.describe_cell(cell), "a_mark": new or "empty"}))

        if s.state == "MISMATCH":
            # same wrong page as already challenged: stay quiet, stay frozen
            if list(labels) == s.mismatch.get("seen"):
                return effects
            # the page changed while frozen — re-judge it as if we were back
            # in the state the mismatch came from, so a corrected page heals
            # the session by itself (and a differently-wrong one re-freezes
            # with updated details)
            from_state = s.mismatch.get("from", "HUMAN_TURN")
            s.state = from_state if from_state in ("HUMAN_TURN", "AWAIT_DRAW") \
                else "HUMAN_TURN"
            s.expected_cell = (s.mismatch.get("expected_cell")
                               if s.state == "AWAIT_DRAW" else None)
            s.mismatch = None
            return effects + self._resolve_stable(s, labels, conf,
                                                  via_arbiter=via_arbiter)

        if s.state == "AWAIT_DRAW":
            expected = self.game.apply(s.board, s.expected_cell, self.game.agent_mark)
            if list(labels) == expected:
                s.board = expected
                s.history.append((self.game.agent_mark, s.expected_cell))
                s.last_conf = conf
                cell = s.expected_cell
                s.expected_cell = None
                effects += [LogEvent("confirm", {"cell": cell,
                                                 "mark": self.game.agent_mark,
                                                 "conf": conf})]
                return effects + self._after_move(s, announce_point="verified")
            # fast play: the asked-for mark landed AND exactly one legal human
            # reply came with it — natural behavior, not a violation. Confirm
            # both, in order.
            extra = _diff(expected, labels)
            if (labels[s.expected_cell] == self.game.agent_mark
                    and len(extra) == 1 and extra[0][1] == ""
                    and extra[0][2] == self.game.human_mark):
                cell_o, cell_x = s.expected_cell, extra[0][0]
                s.board = list(expected)
                s.history.append((self.game.agent_mark, cell_o))
                s.last_conf = conf
                s.expected_cell = None
                effects += [LogEvent("confirm", {"cell": cell_o,
                                                 "mark": self.game.agent_mark,
                                                 "conf": conf, "fast_play": True})]
                if self.game.winner(s.board) or self.game.is_draw(s.board):
                    return effects + self._after_move(s, announce_point="verified")
                effects.append(Announce("verified"))
                s.board = list(labels)
                s.history.append((self.game.human_mark, cell_x))
                effects += [LogEvent("confirm", {"cell": cell_x,
                                                 "mark": self.game.human_mark,
                                                 "conf": conf, "fast_play": True})]
                return effects + self._after_human_move(s, cell_x)
            self._enter_mismatch(
                s, seen=list(labels),
                detail=f"expected {self.game.agent_mark} in "
                       f"{self.game.describe_cell(s.expected_cell)}")
            # report the OFFENDING change, not whichever diff comes first —
            # the asked-for mark may well be among the changes and correct
            offending = [ch for ch in changed
                         if not (ch[0] == s.mismatch["expected_cell"]
                                 and ch[2] == self.game.agent_mark)] or changed
            return effects + self._mismatch_announce(s, offending)

        # HUMAN_TURN / CONFIRMING
        legal = (len(changed) == 1
                 and changed[0][1] == ""
                 and changed[0][2] == self.game.human_mark)
        if legal:
            cell = changed[0][0]
            s.board = list(labels)
            s.history.append((self.game.human_mark, cell))
            s.last_conf = conf
            effects += [LogEvent("confirm", {"cell": cell,
                                             "mark": self.game.human_mark,
                                             "conf": conf})]
            return effects + self._after_human_move(s, cell)
        detail = _illegal_detail(changed, self.game)
        self._enter_mismatch(s, seen=list(labels), detail=detail)
        return effects + [Announce("illegal", {"detail": detail}),
                          LogEvent("mismatch", {"detail": detail,
                                                "changed": changed})]

    def _after_human_move(self, s: Session, cell: int) -> list:
        effects = [Announce("react", {"cell": self.game.describe_cell(cell)})]
        return effects + self._after_move(s, announce_point=None, human=True)

    def _after_move(self, s: Session, announce_point: str | None,
                    human: bool = False) -> list:
        effects: list = []
        if announce_point:
            effects.append(Announce(announce_point))
        winner = self.game.winner(s.board)
        if winner or self.game.is_draw(s.board):
            s.state = "GAME_OVER"
            s.result = ("draw" if not winner
                        else "human" if winner == self.game.human_mark
                        else "agent")
            effects.append(GameEnded(result=s.result))
            return effects
        if human:
            s.state = "AGENT_TURN"
            effects.append(EngineTurn())
        else:
            s.state = "HUMAN_TURN"
        return effects

    # --- mismatch handling ----------------------------------------------------

    def _enter_mismatch(self, s: Session, seen: list, detail: str) -> None:
        s.mismatch = {"from": s.state if s.state != "MISMATCH" else
                      s.mismatch.get("from", "HUMAN_TURN"),
                      "expected_cell": s.expected_cell,
                      "seen": seen, "detail": detail}
        s.mismatches += 1
        s.state = "MISMATCH"
        s.reset_candidate()

    def _mismatch_announce(self, s: Session, changed: list) -> list:
        cell, _, new = changed[0]
        exp = s.mismatch["expected_cell"]
        log = LogEvent("mismatch", {"expected": exp, "changed": changed})
        if exp is not None and cell == exp:
            # right cell, wrong symbol — "X in top-right, not top-right" is
            # not a sentence anyone should hear
            return [Announce("mismatch_wrong_mark", {
                        "expected": self.game.describe_cell(exp),
                        "a_mark": self.game.agent_mark,
                        "seen_mark": new or "an erased mark"}), log]
        return [Announce("mismatch", {
                    "expected": self.game.describe_cell(exp) if exp is not None
                    else "no change",
                    "seen": self.game.describe_cell(cell),
                    "seen_mark": new or "an erased mark"}), log]

    def _resume_from_mismatch(self, s: Session) -> list:
        from_state = s.mismatch["from"] if s.mismatch else "HUMAN_TURN"
        s.expected_cell = s.mismatch.get("expected_cell") if s.mismatch else None
        s.mismatch = None
        s.state = "AWAIT_DRAW" if from_state == "AWAIT_DRAW" else "HUMAN_TURN"
        s.reset_candidate()
        return [Announce("fixed"), LogEvent("mismatch_resolved", {"via": "page_fixed"})]

    def _accept_reading(self, s: Session) -> list:
        seen = s.mismatch["seen"]
        from_state = s.mismatch["from"]
        expected_cell = s.mismatch.get("expected_cell")
        s.board = list(seen)
        s.mismatch = None
        s.expected_cell = None
        effects: list = [Announce("accepted"),
                         LogEvent("mismatch_resolved", {"via": "accept_reading"})]
        if from_state == "AWAIT_DRAW":
            # wherever the O actually landed is the agent's move now
            agent_cells = [i for i, m in enumerate(seen)
                           if m == self.game.agent_mark
                           and (self.game.agent_mark, i) not in s.history]
            landed = agent_cells[-1] if agent_cells else expected_cell
            if landed is not None:
                s.history.append((self.game.agent_mark, landed))
        # whatever was accepted, mark counts say whose turn it is now
        x = sum(1 for m in s.board if m == self.game.human_mark)
        o = sum(1 for m in s.board if m == self.game.agent_mark)
        winner = self.game.winner(s.board)
        if winner or self.game.is_draw(s.board):
            return effects + self._after_move(s, announce_point=None)
        if x > o:
            s.state = "AGENT_TURN"
            effects.append(EngineTurn())
        else:
            s.state = "HUMAN_TURN"
        return effects


# --- helpers ------------------------------------------------------------------


def _diff(board: list[str], labels: tuple) -> list[tuple[int, str, str]]:
    return [(i, board[i], labels[i]) for i in range(len(board))
            if board[i] != labels[i]]


def _mean_conf(p: PerceptionResult) -> float:
    if not p.cells:
        return 0.0
    return round(sum(c.conf for c in p.cells) / len(p.cells), 2)


def _illegal_detail(changed: list, game) -> str:
    if len(changed) > 1:
        return f"{len(changed)} cells changed at once"
    cell, old, new = changed[0]
    where = game.describe_cell(cell)
    if old != "" and new == "":
        return f"the {old} in the {where} vanished"
    if old != "":
        return f"the {where} was already {old}, now reads {new}"
    if new == game.agent_mark:
        return f"that's my mark ({new}) in the {where} — it's your turn"
    return f"unexpected {new} in the {where}"
