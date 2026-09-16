"""Announcer templates — every line the agent speaks, one file, editable live.

Voice rules: short declarative lines, dry, confident. No exclamation marks,
no filler, no apologizing. 3-4 variants per announce point so two consecutive
games don't sound canned. Rotation-safe early phrasing: cell names come from
game.describe_cell(), which the client board view shares.

Format keys available per point are documented inline.
"""

PHRASES: dict[str, list[str]] = {
    # {conf} percent int
    "lock": [
        "I see the board — confidence {conf}%. You're X, you start. Go ahead.",
        "Board locked at {conf}%. You're X — your move.",
        "Got the grid, confidence {conf}%. Draw your X when ready.",
    ],
    # {conf} percent, {xs}/{os} mark counts, {turn_line} — lock on a game in progress
    "lock_resume": [
        "I see the board — {xs} X and {os} O already played, confidence {conf}%. Resuming. {turn_line}",
        "Board locked at {conf}%, mid-game: {xs} X, {os} O on the page. {turn_line}",
        "Picking up where this page left off — {xs} X, {os} O. {turn_line}",
    ],
    # {cell} name — reaction to the human's confirmed mark
    "react": [
        "{cell_cap} — noted.",
        "X in the {cell}. Noted.",
        "{cell_cap} taken. Bold.",
        "I saw that X land in the {cell}.",
    ],
    # {cell} name — the agent's own move + explicit drawing instruction
    "own_move": [
        "I'll take the {cell}. Draw an O there for me.",
        "My move: {cell}. Please draw my O in that cell.",
        "I play the {cell}. Put an O there and take your hand out of frame.",
    ],
    # await resolved
    "verified": [
        "Got it. Your turn.",
        "That's my O, confirmed. You're up.",
        "Ink matches the call. Your move.",
    ],
    # {expected} / {seen} cell names, {seen_mark}
    "mismatch": [
        "I asked for {expected} but I'm seeing {seen_mark} in {seen}. Did I misread, or are we improvising? Fix the page or tell me to accept it.",
        "The page shows {seen_mark} in {seen}; I expected {expected}. Fix it, or tell me to accept the reading.",
        "That's not what I called — {seen_mark} in {seen}, not {expected}. Your call: fix the page or I accept it.",
    ],
    # {expected} cell, {a_mark} what was asked, {seen_mark} what's there
    "mismatch_wrong_mark": [
        "I asked for {a_mark} in the {expected} but I'm reading {seen_mark} there. Fix it, or tell me to accept the reading.",
        "The {expected} has {seen_mark} where my {a_mark} should be. Redraw it, or tell me to accept.",
        "That's {seen_mark} in the {expected} — I called {a_mark}. Fix the page or say the word.",
    ],
    # illegal human move: {detail}
    "illegal": [
        "That doesn't parse as a legal move — {detail}. Fix the page or tell me to accept what I see.",
        "Hold on: {detail}. Fix it, or tell me to take the camera reading as truth.",
    ],
    # {cell} name — low confidence, arbiter invoked
    "arbiter_look": [
        "Not sure about the {cell} — looking closer.",
        "That {cell} read is shaky. One moment, taking a closer look.",
        "Low confidence on the {cell}. Checking properly.",
    ],
    # {cell}, {mark} — arbiter resolution
    "arbiter_done": [
        "Confirmed — that's {a_mark} in the {cell}.",
        "Closer look says {a_mark}, {cell}. Proceeding.",
        "Resolved: {a_mark} in the {cell}.",
    ],
    "arbiter_fail": [
        "I can't resolve that mark on my own. Fix the page, or tell me to accept my best guess.",
        "Still ambiguous after a closer look. Redraw it thicker, or tell me to accept the reading.",
    ],
    # {result} = "I win" | "You win" | "Draw", {summary} learned line
    "game_over": [
        "{result}. {summary}",
        "{result}. Good game. {summary}",
        "That's the game — {result_lower}. {summary}",
    ],
    "reset": [
        "Fresh board. Point me at a clean page.",
        "Reset. Give me a new grid when you're ready.",
    ],
    "start": [
        "Looking for the board. Hold the camera steady over the page.",
        "Show me the grid — thick strokes, page flat.",
    ],
    "fixed": [
        "Page fixed — back to it.",
        "Clean again. Continuing where we were.",
        "That matches now. Go on.",
    ],
    "accepted": [
        "Accepted. The page is the truth now — moving on.",
        "Fine, we play it as it lies.",
    ],
    "fixing": [
        "Waiting on the page fix. Take your time.",
        "I'll hold while you fix it.",
    ],
}
